/**
 * @file dodecode.cc
 * @brief Implmentation file for the DoDecode class. Currently, just supports
 * basestation
 */
#include "dodecode.h"

#include "concurrent_queue_wrapper.h"
#include "phy_ldpc_decoder_5gnr.h"

static constexpr bool kPrintLLRData = false;
static constexpr bool kPrintDecodedData = false;

static constexpr size_t kVarNodesSize = 1024 * 1024 * sizeof(int16_t);

DoDecode::DoDecode(
    Config* in_config, int in_tid,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers,
    MacScheduler* mac_sched, PhyStats* in_phy_stats, Stats* in_stats_manager)
    : Doer(in_config, in_tid),
      demod_buffers_(demod_buffers),
      decoded_buffers_(decoded_buffers),
      mac_sched_(mac_sched),
      phy_stats_(in_phy_stats),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kDecode, in_tid);
  resp_var_nodes_ = static_cast<int16_t*>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, kVarNodesSize));
}

DoDecode::~DoDecode() { std::free(resp_var_nodes_); }

EventData DoDecode::Launch(size_t tag) {
  const LDPCconfig& ldpc_config = cfg_->LdpcConfig(Direction::kUplink);
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t symbol_idx_ul = cfg_->Frame().GetULSymbolIdx(symbol_id);
  const size_t cb_id = gen_tag_t(tag).cb_id_;
  const size_t symbol_offset =
      cfg_->GetTotalDataSymbolIdxUl(frame_id, symbol_idx_ul);
  const size_t cur_cb_id = (cb_id % ldpc_config.NumBlocksInSymbol());
  const size_t sched_ue_id = (cb_id / ldpc_config.NumBlocksInSymbol());
  const size_t ue_id = mac_sched_->ScheduledUeIndex(frame_id, 0, sched_ue_id);
  const size_t frame_slot = (frame_id % kFrameWnd);
  const size_t num_bytes_per_cb = cfg_->NumBytesPerCb(Direction::kUplink);
  if (kDebugPrintInTask == true) {
    std::printf(
        "In doDecode thread %d: frame: %zu, symbol: %zu, code block: "
        "%zu, ue: %zu offset %zu\n",
        tid_, frame_id, symbol_id, cur_cb_id, ue_id, symbol_offset);
  }

  size_t start_tsc = GetTime::WorkerRdtsc();

  struct bblib_ldpc_decoder_5gnr_request ldpc_decoder_5gnr_request {};
  struct bblib_ldpc_decoder_5gnr_response ldpc_decoder_5gnr_response {};

  // Decoder setup
  int16_t num_filler_bits = 0;
  int16_t num_channel_llrs = ldpc_config.NumCbCodewLen();

  ldpc_decoder_5gnr_request.numChannelLlrs = num_channel_llrs;
  ldpc_decoder_5gnr_request.numFillerBits = num_filler_bits;
  ldpc_decoder_5gnr_request.maxIterations = ldpc_config.MaxDecoderIter();
  ldpc_decoder_5gnr_request.enableEarlyTermination =
      ldpc_config.EarlyTermination();
  ldpc_decoder_5gnr_request.Zc = ldpc_config.ExpansionFactor();
  ldpc_decoder_5gnr_request.baseGraph = ldpc_config.BaseGraph();
  ldpc_decoder_5gnr_request.nRows = ldpc_config.NumRows();

  int num_msg_bits = ldpc_config.NumCbLen() - num_filler_bits;
  ldpc_decoder_5gnr_response.numMsgBits = num_msg_bits;
  ldpc_decoder_5gnr_response.varNodes = resp_var_nodes_;

  int8_t* llr_buffer_ptr =
      demod_buffers_[frame_slot][symbol_idx_ul][sched_ue_id] +
      (cfg_->ModOrderBits(Direction::kUplink) *
       (ldpc_config.NumCbCodewLen() * cur_cb_id));

  uint8_t* decoded_buffer_ptr =
      (uint8_t*)decoded_buffers_[frame_slot][symbol_idx_ul][ue_id] +
      (cur_cb_id * Roundup<64>(num_bytes_per_cb));

  ldpc_decoder_5gnr_request.varNodes = llr_buffer_ptr;
  ldpc_decoder_5gnr_response.compactedMessageBytes = decoded_buffer_ptr;

  size_t start_tsc1 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;

  bblib_ldpc_decoder_5gnr(&ldpc_decoder_5gnr_request,
                          &ldpc_decoder_5gnr_response);

  if (cfg_->ScrambleEnabled()) {
    scrambler_->Descramble(decoded_buffer_ptr, num_bytes_per_cb);
  }

  size_t start_tsc2 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

  // if (kPrintLLRData) {
  //   std::printf("LLR data, symbol_offset: %zu\n", symbol_offset);
  //   for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
  //     std::printf("%d ", *(llr_buffer_ptr + i));
  //   }
  //   std::printf("\n");
  // }

  if (kPrintLLRData) {
    std::printf("LLR data, symbol_offset: %zu\n", symbol_offset);
    // Cast llr_buffer_ptr to uint8_t* to treat the data as bytes
    uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(llr_buffer_ptr);
    size_t num_bytes = ldpc_config.NumCbCodewLen();
    size_t num_32bit_values = num_bytes / 4;
    size_t remaining_bytes = num_bytes % 4;

    for (size_t i = 0; i < num_32bit_values; i++) {
      // Read 4 bytes and convert them into a 32-bit integer
      uint32_t value = 0;
      for (size_t j = 0; j < 4; j++) {
        value |= static_cast<uint32_t>(byte_ptr[i * 4 + j]) << (j * 8);
      }
      if (i > 0) {
        std::printf(", ");  // Separate 32-bit values with a comma
      }
      std::printf("0x%08x",
                  value);  // Print the 32-bit value in hexadecimal format
    }

    // Print remaining bytes if any
    if (remaining_bytes > 0) {
      if (num_32bit_values > 0) {
        std::printf(", ");
      }
      uint32_t value = 0;
      for (size_t j = 0; j < remaining_bytes; j++) {
        value |= static_cast<uint32_t>(byte_ptr[num_32bit_values * 4 + j])
                 << (j * 8);
      }
      std::printf("0x%08x",
                  value);  // Print the remaining bytes as a 32-bit value
    }

    std::printf("\n");
  }

  // if (kPrintDecodedData) {
  //   std::printf("Decoded data\n");
  //   for (size_t i = 0; i < (ldpc_config.NumCbLen() >> 3); i++) {
  //     std::printf("%u ", *(decoded_buffer_ptr + i));
  //   }
  //   std::printf("\n");
  // }

  if (kPrintDecodedData) {
    std::printf("Decoded data\n");
    // Cast decoded_buffer_ptr to uint8_t* to treat the data as bytes
    uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(decoded_buffer_ptr);
    size_t num_bytes = (ldpc_config.NumCbLen() >> 3);  // Number of bytes
    size_t num_32bit_values = num_bytes / 4;
    size_t remaining_bytes = num_bytes % 4;

    for (size_t i = 0; i < num_32bit_values; i++) {
      // Read 4 bytes and convert them into a 32-bit integer
      uint32_t value = 0;
      for (size_t j = 0; j < 4; j++) {
        value |= static_cast<uint32_t>(byte_ptr[i * 4 + j]) << (j * 8);
      }
      if (i > 0) {
        std::printf(", ");  // Separate 32-bit values with a comma
      }
      std::printf("0x%08x",
                  value);  // Print the 32-bit value in hexadecimal format
    }

    // Print remaining bytes if any
    if (remaining_bytes > 0) {
      if (num_32bit_values > 0) {
        std::printf(", ");
      }
      uint32_t value = 0;
      for (size_t j = 0; j < remaining_bytes; j++) {
        value |= static_cast<uint32_t>(byte_ptr[num_32bit_values * 4 + j])
                 << (j * 8);
      }
      std::printf("0x%08x",
                  value);  // Print the remaining bytes as a 32-bit value
    }

    std::printf("\n");
  }

  if ((kEnableMac == false) && (kPrintPhyStats == true) &&
      (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
    phy_stats_->UpdateDecodedBits(ue_id, symbol_offset, frame_slot,
                                  num_bytes_per_cb * 8);
    phy_stats_->IncrementDecodedBlocks(ue_id, symbol_offset, frame_slot);
    size_t block_error(0);
    for (size_t i = 0; i < num_bytes_per_cb; i++) {
      uint8_t rx_byte = decoded_buffer_ptr[i];
      auto tx_byte = static_cast<uint8_t>(
          cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, symbol_idx_ul,
                            ue_id, cur_cb_id)[i]);
      phy_stats_->UpdateBitErrors(ue_id, symbol_offset, frame_slot, tx_byte,
                                  rx_byte);
      if (rx_byte != tx_byte) {
        block_error++;
      }
    }
    phy_stats_->UpdateBlockErrors(ue_id, symbol_offset, frame_slot,
                                  block_error);
  }
  size_t end = GetTime::WorkerRdtsc();
  size_t duration_3 = end - start_tsc2;
  // AGORA_LOG_INFO("Cell # %d: \tframe %d, BLER memcmp check time is: %.4f ms\n", (int)cfg_->bbdev_id_1(), frame_id, GetTime::CyclesToUs(duration_3, cfg_->FreqGhz()*1000));

  size_t duration = GetTime::WorkerRdtsc() - start_tsc;
  duration_stat_->task_duration_[0] += duration;
  duration_stat_->task_count_++;
  if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
    std::printf("Thread %d Decode takes %.2f\n", tid_,
                GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
  }

  return EventData(EventType::kDecode, tag);
}
