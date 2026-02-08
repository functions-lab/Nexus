/**
 * @file dodecode_acc.cc
 * @brief Implmentation file for the DoDecode class with ACC100 acceleration. 
 */

#include "dodecode_acc.h"

#include "concurrent_queue_wrapper.h"
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"

#define NUM_VF 4

#define GET_SOCKET(socket_id) (((socket_id) == SOCKET_ID_ANY) ? 0 : (socket_id))
#define MAX_RX_BYTE_SIZE 1500
#define CACHE_SIZE 128
#define NUM_QUEUES 4
#define LCORE_ID 36
#define NUM_ELEMENTS_IN_POOL 2047
#define NUM_ELEMENTS_IN_MEMPOOL 16383
#define DATA_ROOM_SIZE 45488
static constexpr bool kPrintLLRData = false;
static constexpr bool kPrintDecodedData = false;
static constexpr bool kPrintACC100Byte = false;
static constexpr bool kPrintTxByte = false;
static constexpr bool kPrintMbufData = false;
static constexpr bool kMubfLLRCheck = false;

#include <immintrin.h>

int memcmp_avx2(const uint8_t* data1, const uint8_t* data2, size_t num_bytes) {
    size_t num_blocks = num_bytes / 32;  // Process 32 bytes at a time for AVX2
    for (size_t i = 0; i < num_blocks; i++) {
        __m256i chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data1 + i * 32));
        __m256i chunk2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data2 + i * 32));
        __m256i result = _mm256_xor_si256(chunk1, chunk2);  // XOR each byte, 0 if equal
        if (!_mm256_testz_si256(result, result)) {  // Check if any byte differs
            return -1;
        }
    }
    // Compare any remaining bytes
    size_t remainder = num_bytes % 32;
    for (size_t i = num_bytes - remainder; i < num_bytes; i++) {
        if (data1[i] != data2[i]) return -1;
    }
    return 0;
}

static void
ldpc_input_llr_scaling(struct rte_bbdev_op_data *input_ops,  const int8_t llr_size,
		const int8_t llr_decimals)
{
	if (input_ops == NULL)
		return;

	uint16_t i = 0, byte_idx;
	int16_t llr_max, llr_min, llr_tmp;
  // std::cout << "in ldpc_input_llr_scaling input_ops is not null !!!!!!!!!\n";
	llr_max = (1 << (llr_size - 1)) - 1;
	llr_min = -llr_max;
		struct rte_mbuf *m = input_ops[i].data;
		while (m != NULL) {
			int8_t *llr = rte_pktmbuf_mtod_offset(m, int8_t *,
					input_ops[i].offset);
			for (byte_idx = 0; byte_idx < rte_pktmbuf_data_len(m);
					++byte_idx) {

				llr_tmp = llr[byte_idx];
				if (llr_decimals == 4)
					llr_tmp *= 8;
				else if (llr_decimals == 2)
					llr_tmp *= 2;
				else if (llr_decimals == 0)
					llr_tmp /= 2;
				llr_tmp = RTE_MIN(llr_max,
						RTE_MAX(llr_min, llr_tmp));
				llr[byte_idx] = (int8_t) llr_tmp;
			}

			m = m->next;
		}
}

static constexpr size_t kVarNodesSize = 1024 * 1024 * sizeof(int16_t);

static unsigned int optimal_mempool_size(unsigned int val) {
  return rte_align32pow2(val + 1) - 1;
}

void print_uint32(const uint32_t* array, size_t totalByteLength) {
  size_t totalWordLength = (totalByteLength + 3) / 4;

  for (int i = 0; i < (int)totalWordLength; i++) {
    // Extract and print the byte
    printf("%08X ", array[i]);
  }
}

void print_casted_uint8_hex(const uint8_t* array, size_t totalByteLength) {
  for (size_t i = 0; i < totalByteLength; i++) {
    // Cast each int8_t to uint8_t and print as two-digit hexadecimal
    printf("%02X ", (uint8_t)array[i]);
  }
  printf("\n");  // Add a newline for readability
}

DoDecode_ACC::DoDecode_ACC(
    Config *in_config, int in_tid,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &demod_buffers,
    // PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, uint32_t> &llr_buffers,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &decoded_buffers,
    PhyStats *in_phy_stats, Stats *in_stats_manager)
    : Doer(in_config, in_tid),
      demod_buffers_(demod_buffers),
      // llr_buffers_(llr_buffers),
      decoded_buffers_(decoded_buffers),
      phy_stats_(in_phy_stats),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kDecode, in_tid);
  resp_var_nodes_ = static_cast<int16_t *>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, kVarNodesSize));
  std::string core_list = std::to_string(LCORE_ID);  // this is hard set to core 36
  const size_t num_ul_syms = cfg_->Frame().NumULSyms(); 
  const size_t num_ue = cfg_->UeAntNum();

  const char *rte_argv[] = {"txrx",        "-l",           core_list.c_str(), "-a", "18:00.0", "-a", "18:00.1", "-a", "18:00.2", "-a", "18:00.3",
                            "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  int ret = rte_eal_init(rte_argc, const_cast<char **>(rte_argv));
  int ret_2;
  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  std::cout << "num bbdevs: " << nb_bbdevs << std::endl;

  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");
  struct rte_bbdev_info info[NUM_VF];

  for (int i=0; i<nb_bbdevs; i++){
    dev_ids.push_back(i);
  }

  for (int i=0; i<nb_bbdevs; i++){
    rte_bbdev_intr_enable(dev_ids[i]);
    rte_bbdev_info_get(dev_ids[i], &info[i]);
  }

  if (ret < 0) {
    printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_ids[0], NUM_QUEUES,
           rte_socket_id(), ret);
  }

  std::vector <int> socket_ids;
  for (int i = 0; i < NUM_VF; i++){
    socket_ids.push_back(GET_SOCKET(info[i].socket_id));
  }

  for (int i = 0; i < NUM_VF; i++) {
    ret = rte_bbdev_setup_queues(i, NUM_QUEUES, (int)socket_ids[i]);
  }

  struct rte_bbdev_queue_conf qconf[NUM_VF];
  for (int i = 0; i < NUM_VF; i++){
    qconf[i].socket = info[i].socket_id;
    qconf[i].queue_size = info[i].drv.queue_size_lim;
    qconf[i].op_type = RTE_BBDEV_OP_LDPC_DEC;
    qconf[i].priority = 0;
  }

  for (int i = 0; i < NUM_VF; i++){
    for (int q_id = 0; q_id < NUM_QUEUES; q_id++) {
      /* Configure all queues belonging to this bbdev device */
      ret = rte_bbdev_queue_configure(dev_ids[i], q_id, &qconf[i]);
      if (ret < 0)
        rte_exit(EXIT_FAILURE,
                "ERROR(%d): BBDEV %u queue %u not configured properly\n", ret,
                dev_ids[i], q_id);
    }
  }

  for (int i = 0; i < NUM_VF; i++){
    ret = rte_bbdev_start(dev_ids[i]);
  }

  for (int i = 0; i < NUM_VF; i++){
    std::string pool_name = "LDPC_DEC_poo_" + std::to_string(i);
    ops_mp[i] = rte_bbdev_op_pool_create(pool_name.c_str(),
                                    RTE_BBDEV_OP_LDPC_DEC, NUM_ELEMENTS_IN_POOL, OPS_CACHE_SIZE,
                                    (int)socket_ids[i]);
    if (ops_mp[i] == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations."
                << std::endl;
    } else {
      std::cout << "Memory pool for bbdev operations created successfully."
                << std::endl;
    }
  }

  for (int i = 0; i < NUM_VF; i++){
    std::string pool_name_in = "in_pool_" + std::to_string(i);
    std::string pool_name_out = "hard_out_pool_pool_" + std::to_string(i);

    in_mbuf_pool[i] = rte_pktmbuf_pool_create(pool_name_in.c_str(), NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, (int)socket_ids[i]);
    out_mbuf_pool[i] =
      rte_pktmbuf_pool_create(pool_name_out.c_str(), NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, (int)socket_ids[i]);
    if (in_mbuf_pool[i] == nullptr or out_mbuf_pool[i] == nullptr) {
      std::cerr << "Error: Unable to create mbuf pool: "
                << rte_strerror(rte_errno) << std::endl;
    }
  }

  int rte_alloc_ref;
  for (int i = 0; i < NUM_VF; i++){
    rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp[i], ref_dec_op[i], num_ul_syms * num_ue);
    if (rte_alloc_ref != TEST_SUCCESS ) {
      rte_exit(EXIT_FAILURE, "Failed to alloc bulk\n");
    }
  }

  std::cout << "rte_bbdev_dec_op_alloc_bulk successful\n";

  const struct rte_bbdev_op_cap *capabilities;
  for (int i = 0; i < NUM_VF; i++){
    const struct rte_bbdev_op_cap *cap = info[i].drv.capabilities;
    // *capabilities = NULL;
    rte_bbdev_info_get(dev_ids[i], &info[i]);
    for (unsigned int j = 0; cap->type != RTE_BBDEV_OP_NONE; ++j, ++cap) {
      std::cout << "cap is: " << cap->type << std::endl;
      if (cap->type == RTE_BBDEV_OP_LDPC_DEC) {
        capabilities = cap;
        std::cout << "capability is being set to: " << capabilities->type
                  << std::endl;
        std::cout << std::endl;
        break;
      }
    }
  }
      
  inputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));
  hard_outputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));

  int ret_socket_in = allocate_buffers_on_socket(
      inputs, 1 * sizeof(struct rte_bbdev_op_data), 0);
  int ret_socket_hard_out = allocate_buffers_on_socket(
      hard_outputs, 1 * sizeof(struct rte_bbdev_op_data), 1);

  if (ret_socket_in != TEST_SUCCESS || ret_socket_hard_out != TEST_SUCCESS) {
    rte_exit(EXIT_FAILURE, "Failed to allocate socket\n");
  }

  ldpc_llr_decimals = capabilities->cap.ldpc_dec.llr_decimals;
  ldpc_llr_size = capabilities->cap.ldpc_dec.llr_size;
  ldpc_cap_flags = capabilities->cap.ldpc_dec.capability_flags;

  min_alignment = info[0].drv.min_alignment;

  const LDPCconfig &ldpc_config = cfg_->LdpcConfig(Direction::kUplink);

  int iter_num = num_ul_syms * num_ue;
  q_m = cfg_->ModOrderBits(Direction::kUplink);
  e = ldpc_config.NumCbCodewLen();

  for (int i = 0; i < iter_num; i++) {
    for (int j = 0; j < NUM_VF; j++){
    // ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
      ref_dec_op[j][i]->ldpc_dec.basegraph = (uint8_t)ldpc_config.BaseGraph();
      ref_dec_op[j][i]->ldpc_dec.z_c = (uint16_t)ldpc_config.ExpansionFactor();
      ref_dec_op[j][i]->ldpc_dec.n_filler = (uint16_t)0;
      ref_dec_op[j][i]->ldpc_dec.rv_index = (uint8_t)0;
      ref_dec_op[j][i]->ldpc_dec.n_cb = (uint16_t)ldpc_config.NumCbCodewLen();
      // std::cout << "n_cb is: " << (uint16_t)ldpc_config.NumCbCodewLen() <<  std::endl;
      ref_dec_op[j][i]->ldpc_dec.q_m = (uint8_t)q_m;
      ref_dec_op[j][i]->ldpc_dec.code_block_mode = (uint8_t)1;
      ref_dec_op[j][i]->ldpc_dec.cb_params.e = (uint32_t)e;
      // std::cout << "e is: " << (uint32_t)e <<  std::endl;
      if (!check_bit(ref_dec_op[j][i]->ldpc_dec.op_flags,
                    RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)) {
        ref_dec_op[j][i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
      }
      ref_dec_op[j][i]->ldpc_dec.iter_max = (uint8_t)ldpc_config.MaxDecoderIter();
      ref_dec_op[j][i]->opaque_data = (void *)(uintptr_t)i;
    }
  }
  std::cout << "" << std::endl;
  AGORA_LOG_INFO("rte_pktmbuf_alloc successful\n");
}

DoDecode_ACC::~DoDecode_ACC() {
  std::free(resp_var_nodes_);
}

int DoDecode_ACC::allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers,
                                             const int len, const int socket) {
  int i;
  *buffers = static_cast<struct rte_bbdev_op_data *>(
      rte_zmalloc_socket(NULL, len, 0, socket));
  if (*buffers == NULL) {
    printf("WARNING: Failed to allocate op_data on socket %d\n", socket);
    /* try to allocate memory on other detected sockets */
    for (i = 0; i < socket; i++) {
      *buffers = static_cast<struct rte_bbdev_op_data *>(
          rte_zmalloc_socket(NULL, len, 0, i));
      if (*buffers != NULL) break;
    }
  }
  return (*buffers == NULL) ? TEST_FAILED : TEST_SUCCESS;
}

EventData DoDecode_ACC::Launch(size_t tag) {
  const LDPCconfig &ldpc_config = cfg_->LdpcConfig(Direction::kUplink);
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t symbol_idx_ul = cfg_->Frame().GetULSymbolIdx(symbol_id);
  const size_t num_ul_syms = cfg_->Frame().NumULSyms();  
  const size_t cb_id = gen_tag_t(tag).cb_id_;
  const size_t symbol_offset =
      cfg_->GetTotalDataSymbolIdxUl(frame_id, symbol_idx_ul);
  const size_t cur_cb_id = (cb_id % ldpc_config.NumBlocksInSymbol());
  const size_t ue_id = (cb_id / ldpc_config.NumBlocksInSymbol());
  const size_t num_ue = cfg_->UeAntNum();
  const size_t frame_slot = (frame_id % kFrameWnd);
  const size_t num_bytes_per_cb = cfg_->NumBytesPerCb(Direction::kUplink);
  if (kDebugPrintInTask == true) {
    std::printf(
        "In doDecode thread %d: frame: %zu, symbol: %zu, code block: "
        "%zu, ue: %zu offset %zu\n",
        tid_, frame_id, symbol_id, cur_cb_id, ue_id, symbol_offset);
  }
  
  int total = enq + enq_2 + enq_3 + enq_4;
  // std::cout <<"enq is " << enq << " enq_2 is " << enq_2 << " enq_3 is " << enq_3 << std::endl;
  if (total == num_ul_syms * num_ue - 1) {
    size_t start_tsc = GetTime::WorkerRdtsc();

    int8_t *llr_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;

    llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                            (cfg_->ModOrderBits(Direction::kUplink) *
                              (ldpc_config.NumCbCodewLen() * cur_cb_id));


    if (symbol_idx_ul == 11 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
        std::printf("LLR data, symbol_offset: %zu\n", symbol_offset);
        for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
          std::printf("%04X ", *(llr_buffer_ptr + i));
        }
        std::printf("\n");
    }

    char *data;
    struct rte_bbdev_op_data *bufs = *inputs;
    size_t start_tsc1;

    if (enq_2 != 4 * num_ue) {
      m_head = rte_pktmbuf_alloc(in_mbuf_pool[1]);

      bufs[0].data = m_head;
      bufs[0].offset = 0;
      bufs[0].length = 0;

      data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

      // Copy data from demod_data to the mbuf
      rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                ldpc_config.NumCbCodewLen());
      bufs[0].length += ldpc_config.NumCbCodewLen();

      rte_bbdev_op_data *bufs_out = *hard_outputs;
      m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[1]);

      bufs_out[0].data = m_head_out;
      bufs_out[0].offset = 0;
      bufs_out[0].length = 0;
      // BUG: This line causes a irregular stop of the program when fft_size = 4096,
      //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
      //      sampling rate.
      bufs_out[0].length += ldpc_config.NumCbCodewLen();

      // enq_index = symbol_idx_ul;

      ref_dec_op[1][enq_index_2]->ldpc_dec.input = *inputs[0];
      ref_dec_op[1][enq_index_2]->ldpc_dec.hard_output = *hard_outputs[0];

      rte_pktmbuf_free(m_head);
      rte_pktmbuf_free(m_head_out);

      start_tsc1 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
      
      enq_2 += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[1], 0, &ref_dec_op[1][enq_2], 1);
    } else if (enq != 4 * num_ue) {
      m_head = rte_pktmbuf_alloc(in_mbuf_pool[0]);

      bufs[0].data = m_head;
      bufs[0].offset = 0;
      bufs[0].length = 0;

      data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

      // Copy data from demod_data to the mbuf
      rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                ldpc_config.NumCbCodewLen());
      bufs[0].length += ldpc_config.NumCbCodewLen();

      rte_bbdev_op_data *bufs_out = *hard_outputs;
      m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[0]);

      bufs_out[0].data = m_head_out;
      bufs_out[0].offset = 0;
      bufs_out[0].length = 0;
      // BUG: This line causes a irregular stop of the program when fft_size = 4096,
      //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
      //      sampling rate.
      bufs_out[0].length += ldpc_config.NumCbCodewLen();

      // enq_index = symbol_idx_ul;

      ref_dec_op[0][enq_index]->ldpc_dec.input = *inputs[0];
      ref_dec_op[0][enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

      rte_pktmbuf_free(m_head);
      rte_pktmbuf_free(m_head_out);

      start_tsc1 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
      
      enq += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[0], 0, &ref_dec_op[0][enq], 1);
    } else if (enq_3 != 4 * num_ue){
      m_head = rte_pktmbuf_alloc(in_mbuf_pool[0]);

      bufs[0].data = m_head;
      bufs[0].offset = 0;
      bufs[0].length = 0;

      data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

      // Copy data from demod_data to the mbuf
      rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                ldpc_config.NumCbCodewLen());
      bufs[0].length += ldpc_config.NumCbCodewLen();

      rte_bbdev_op_data *bufs_out = *hard_outputs;
      m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[0]);

      bufs_out[0].data = m_head_out;
      bufs_out[0].offset = 0;
      bufs_out[0].length = 0;
      // BUG: This line causes a irregular stop of the program when fft_size = 4096,
      //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
      //      sampling rate.
      bufs_out[0].length += ldpc_config.NumCbCodewLen();

      ref_dec_op[2][enq_index_3]->ldpc_dec.input = *inputs[0];
      ref_dec_op[2][enq_index_3]->ldpc_dec.hard_output = *hard_outputs[0];

      rte_pktmbuf_free(m_head);
      rte_pktmbuf_free(m_head_out);

      start_tsc1 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
      
      enq_3 += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[2], 0, &ref_dec_op[2][enq_3], 1);
    } else {
      m_head = rte_pktmbuf_alloc(in_mbuf_pool[0]);

      bufs[0].data = m_head;
      bufs[0].offset = 0;
      bufs[0].length = 0;

      data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

      // Copy data from demod_data to the mbuf
      rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                ldpc_config.NumCbCodewLen());
      bufs[0].length += ldpc_config.NumCbCodewLen();

      rte_bbdev_op_data *bufs_out = *hard_outputs;
      m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[0]);

      bufs_out[0].data = m_head_out;
      bufs_out[0].offset = 0;
      bufs_out[0].length = 0;
      // BUG: This line causes a irregular stop of the program when fft_size = 4096,
      //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
      //      sampling rate.
      bufs_out[0].length += ldpc_config.NumCbCodewLen();

      ref_dec_op[3][enq_index_4]->ldpc_dec.input = *inputs[0];
      ref_dec_op[3][enq_index_4]->ldpc_dec.hard_output = *hard_outputs[0];

      rte_pktmbuf_free(m_head);
      rte_pktmbuf_free(m_head_out);

      start_tsc1 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
      
      enq_4 += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[3], 0, &ref_dec_op[3][enq_4], 1);
    }

    // std::cout <<"enq is " << enq << " enq_2 is " << enq_2 << std::endl;
    int retry_count = 0;
    while (deq < enq && retry_count < MAX_DEQUEUE_TRIAL) {
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[0][deq], enq - deq);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100 VF1: enq = %d, deq = %d\n", enq, deq);

    retry_count = 0;
    while (deq_2 < enq_2 && retry_count < MAX_DEQUEUE_TRIAL) {
      deq_2 += rte_bbdev_dequeue_ldpc_dec_ops(1, 0, &ops_deq[1][deq_2], enq_2 - deq_2);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100 VF2: enq = %d, deq = %d\n", enq_2, deq_2);

    retry_count = 0;
    while (deq_3 < enq_3 && retry_count < MAX_DEQUEUE_TRIAL) {
      deq_3 += rte_bbdev_dequeue_ldpc_dec_ops(2, 0, &ops_deq[2][deq_3], enq_3 - deq_3);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100 VF3: enq = %d, deq = %d\n", enq_3, deq_3);

    retry_count = 0;
    while (deq_4 < enq_4 && retry_count < MAX_DEQUEUE_TRIAL) {
      deq_4 += rte_bbdev_dequeue_ldpc_dec_ops(3, 0, &ops_deq[3][deq_4], enq_4 - deq_4);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100 VF4: enq = %d, deq = %d\n", enq_4, deq_4);

    enq = 0;
    deq = 0;
    enq_index = 0;

    enq_2 = 0;
    deq_2 = 0;
    enq_index_2 = 0;

    enq_3 = 0;
    deq_3 = 0;
    enq_index_3 = 0;

    enq_4 = 0;
    deq_4 = 0;
    enq_index_4 = 0;

    size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

    int8_t* ref_byte;

    if ((kEnableMac == false) && (kPrintPhyStats == true) &&
        (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
      // get mbuf from the ops_deq

      struct rte_bbdev_op_ldpc_dec *ops_td;
      unsigned int i = 0;
      unsigned int i_2 = 0;
      unsigned int i_3 = 0;
      unsigned int i_4 = 0;
      struct rte_bbdev_op_data *hard_output;
      struct rte_mbuf *temp_m;
      std::vector<size_t> block_errors(num_ue, 0);
      int ref_symbol_idx;
      int ref_ue_idx;
      // size_t offset = 0;
      
    for (size_t temp_idx = 0; temp_idx < 4 * num_ue; temp_idx++){
      bool print_frame_info = kPrintACC100Byte && frame_id > 100 && frame_id % 100 == 0;

      // for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){  
          ops_td = &ops_deq[0][i]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);
          
          // If temp_m contains multiple bytes, use memcmp to compare
          if (cfg_->ScrambleEnabled()) {
            scrambler_->Descramble(temp_data, num_bytes_per_cb);
          }

          ref_symbol_idx = (int)temp_data[2] + 2;
          ref_ue_idx = (int)temp_data[4];

          
          if (ref_symbol_idx < num_ul_syms) {
            ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, ref_symbol_idx,
                              ref_ue_idx, cur_cb_id);
            if (print_frame_info) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n" << "Content of the CB (in uint32):\n";
              print_casted_uint8_hex(temp_data, num_bytes_per_cb);
              std::cout << "\n\nContent of the ref_bytes are:\n";
              print_casted_uint8_hex(reinterpret_cast<const uint8_t*>(ref_byte), num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
            size_t memcmp_ret = memcmp_avx2(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
            if (memcmp_ret != 0) {
                // Data matches, do something
                block_errors[ref_ue_idx]++;                
            }
          } else {error_1++;}
              i++;
        // }
      }
 
    for (size_t temp_idx = 0; temp_idx < 4 * num_ue; temp_idx++){
      bool print_frame_info = kPrintACC100Byte && frame_id > 100 && frame_id % 100 == 0;

      // for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){  
          ops_td = &ops_deq[1][i_2]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);
          
          // If temp_m contains multiple bytes, use memcmp to compare
          if (cfg_->ScrambleEnabled()) {
            scrambler_->Descramble(temp_data, num_bytes_per_cb);
          }

          ref_symbol_idx = (int)temp_data[2] + 2;
          ref_ue_idx = (int)temp_data[4];

          
          if (ref_symbol_idx < num_ul_syms) {
            ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, ref_symbol_idx,
                              ref_ue_idx, cur_cb_id);
            if (print_frame_info) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n" << "Content of the CB (in uint32):\n";
              print_casted_uint8_hex(temp_data, num_bytes_per_cb);
              std::cout << "\n\nContent of the ref_bytes are:\n";
              print_casted_uint8_hex(reinterpret_cast<const uint8_t*>(ref_byte), num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
            size_t memcmp_ret = memcmp_avx2(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
            if (memcmp_ret != 0) {
                // Data matches, do something
                block_errors[ref_ue_idx]++;                
            }
          } else {error_2++;}
              i_2++;
        // }
      }

      for (size_t temp_idx = 0; temp_idx < 4 * num_ue; temp_idx++){
      bool print_frame_info = kPrintACC100Byte && frame_id > 100 && frame_id % 100 == 0;

      // for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){  
          ops_td = &ops_deq[2][i_3]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);
          
          // If temp_m contains multiple bytes, use memcmp to compare
          if (cfg_->ScrambleEnabled()) {
            scrambler_->Descramble(temp_data, num_bytes_per_cb);
          }

          ref_symbol_idx = (int)temp_data[2] + 2;
          ref_ue_idx = (int)temp_data[4];

          
          if (ref_symbol_idx < num_ul_syms) {
            ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, ref_symbol_idx,
                              ref_ue_idx, cur_cb_id);
            if (print_frame_info) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n" << "Content of the CB (in uint32):\n";
              print_casted_uint8_hex(temp_data, num_bytes_per_cb);
              std::cout << "\n\nContent of the ref_bytes are:\n";
              print_casted_uint8_hex(reinterpret_cast<const uint8_t*>(ref_byte), num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
            size_t memcmp_ret = memcmp_avx2(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
            if (memcmp_ret != 0) {
                // Data matches, do something
                block_errors[ref_ue_idx]++;                
            }
          } else {error_3++;}
              i_3++;
        // }
      }

      for (size_t temp_idx = 0; temp_idx < 4 * num_ue; temp_idx++){
      bool print_frame_info = kPrintACC100Byte && frame_id > 100 && frame_id % 100 == 0;

      // for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){  
          ops_td = &ops_deq[3][i_4]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);
          
          // If temp_m contains multiple bytes, use memcmp to compare
          if (cfg_->ScrambleEnabled()) {
            scrambler_->Descramble(temp_data, num_bytes_per_cb);
          }

          ref_symbol_idx = (int)temp_data[2] + 2;
          ref_ue_idx = (int)temp_data[4];

          
          if (ref_symbol_idx < num_ul_syms) {
            ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, ref_symbol_idx,
                              ref_ue_idx, cur_cb_id);
            if (print_frame_info) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n" << "Content of the CB (in uint32):\n";
              print_casted_uint8_hex(temp_data, num_bytes_per_cb);
              std::cout << "\n\nContent of the ref_bytes are:\n";
              print_casted_uint8_hex(reinterpret_cast<const uint8_t*>(ref_byte), num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
            size_t memcmp_ret = memcmp_avx2(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
            if (memcmp_ret != 0) {
                // Data matches, do something
                block_errors[ref_ue_idx]++;                
            }
          } else {error_4++;}
              i_4++;
        // }
      }

    for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++) {
      phy_stats_->UpdateDecodedBits(temp_ue_id, symbol_offset, frame_slot, num_bytes_per_cb * 8);
      phy_stats_->IncrementDecodedBlocks(temp_ue_id, symbol_offset, frame_slot);
      phy_stats_->UpdateBlockErrors(temp_ue_id, symbol_offset, frame_slot, block_errors[temp_ue_id]);
    }
    }

    size_t end = GetTime::WorkerRdtsc();
    size_t duration_3 = end - start_tsc2;
    size_t duration = end - start_tsc;

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    duration_stat_->task_duration_[3] += duration_3;
    duration_stat_->task_duration_[0] += duration;
    // duration_stat_->task_duration_[0] += 0;

    // duration_stat_->task_count_++;
    if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
      std::printf("Thread %d Decode takes %.2f\n", tid_,
                  GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
    }
    AGORA_LOG_INFO("Error VF1 = %d, VF2 = %d, VF3 = %d, VF4 = %d\n", error_1 - num_ue, error_2 - num_ue, error_3, error_4);
    error_1 = 0;
    error_2 = 0;
    error_3 = 0;
    error_4 = 0;
  } 
  else {
      if (symbol_idx_ul % 4 == 0){
            // std::cout<<"symbol_idx_ul is: " << symbol_idx_ul << ", ue_id is: " << ue_id << std::endl;
        size_t start_tsc_else = GetTime::WorkerRdtsc();

          int8_t *llr_buffer_ptr;
          struct rte_mbuf *m_head;
          struct rte_mbuf *m_head_out;

          llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                                  (cfg_->ModOrderBits(Direction::kUplink) *
                                    (ldpc_config.NumCbCodewLen() * cur_cb_id));     

          if (symbol_idx_ul == 14 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
              std::printf("Hex: LLR data, symbol_offset, symbol_idx is: %zu\n", symbol_idx_ul);
              for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
                  // Cast to uint8_t to ensure correct printing of the byte in hexadecimal
                  std::printf("%02X ", static_cast<uint8_t>(*(llr_buffer_ptr + i)));
              }
              std::printf("\n");
          }
      
          char *data;
          struct rte_bbdev_op_data *bufs = *inputs;

          m_head = rte_pktmbuf_alloc(in_mbuf_pool[0]);

          bufs[0].data = m_head;
          bufs[0].offset = 0;
          bufs[0].length = 0;

          data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

              // Copy data from demod_data to the mbuf
          rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                    ldpc_config.NumCbCodewLen());
          bufs[0].length += ldpc_config.NumCbCodewLen();

          rte_bbdev_op_data *bufs_out = *hard_outputs;
          m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[0]);

          bufs_out[0].data = m_head_out;
          bufs_out[0].offset = 0;
          bufs_out[0].length = 0;

          // Prepare the mbuf to receive the output data
          // char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
          // assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

          // BUG: This line causes a irregular stop of the program when fft_size = 4096,
          //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
          //      sampling rate.
          // rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());
          bufs_out[0].length += ldpc_config.NumCbCodewLen();

        ref_dec_op[0][enq_index]->ldpc_dec.input = *inputs[0];
        ref_dec_op[0][enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

        rte_pktmbuf_free(m_head);
        rte_pktmbuf_free(m_head_out);

        if (kMubfLLRCheck){
          struct rte_mbuf *mbuf_1 = bufs[0].data;
          uint8_t *mbuf_data = rte_pktmbuf_mtod(mbuf_1, uint8_t *);
          size_t data_length = mbuf_1->data_len;
          
          if (kPrintMbufData) {
            size_t num_bytes = data_length;
            size_t num_32bit_values = num_bytes / 4;
            size_t remaining_bytes = num_bytes % 4;

            for (size_t k = 0; k < num_32bit_values; k++) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; j++) {
                value |= static_cast<uint32_t>(mbuf_data[k * 4 + j]) << (j * 8);
            }
            if (k > 0) {
                std::printf(", ");
            }
            std::printf("0x%08x", value);
            }

            // Handle remaining bytes if any
            if (remaining_bytes > 0) {
                if (num_32bit_values > 0) {
                    std::printf(", ");
                }
                uint32_t value = 0;
                for (size_t j = 0; j < remaining_bytes; j++) {
                    value |= static_cast<uint32_t>(mbuf_data[num_32bit_values * 4 + j]) << (j * 8);
                }
                std::printf("0x%08x", value);
            }

            std::printf("\n");
          }

          if (memcmp(mbuf_data, llr_buffer_ptr, data_length) == 0) {
            std::printf("Data in inputs[%u].data matches llr_buffer_ptr\n", 0 );
          } else {
            std::printf("Data mismatch in inputs[%zu].data\n", 0);

            // Print mismatched bytes
            for (size_t j = 0; j < data_length; ++j) {
                if (mbuf_data[j] != (uint8_t)llr_buffer_ptr[j]) {
                    std::printf("Mismatch at byte %zu: mbuf_data=%02x, llr_buffer_ptr=%02x\n",
                                j, mbuf_data[j], (uint8_t)llr_buffer_ptr[j]);
              }
            }
          }
        }
        enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[0][enq], 1);

        size_t end_else = GetTime::WorkerRdtsc();
        size_t duration_else = end_else - start_tsc_else;

        duration_stat_->task_duration_[0] += duration_else;
        enq_index++;
    }
    else if (symbol_idx_ul % 4 == 1){
            // std::cout<<"symbol_idx_ul is: " << symbol_idx_ul << ", ue_id is: " << ue_id << std::endl;
        size_t start_tsc_else = GetTime::WorkerRdtsc();

        int8_t *llr_buffer_ptr;
        struct rte_mbuf *m_head;
        struct rte_mbuf *m_head_out;

        llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                                (cfg_->ModOrderBits(Direction::kUplink) *
                                  (ldpc_config.NumCbCodewLen() * cur_cb_id));     

        if (symbol_idx_ul == 14 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
            std::printf("Hex: LLR data, symbol_offset, symbol_idx is: %zu\n", symbol_idx_ul);
            for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
                // Cast to uint8_t to ensure correct printing of the byte in hexadecimal
                std::printf("%02X ", static_cast<uint8_t>(*(llr_buffer_ptr + i)));
            }
            std::printf("\n");
        }
    
        char *data;
        struct rte_bbdev_op_data *bufs = *inputs;

        m_head = rte_pktmbuf_alloc(in_mbuf_pool[1]);

        bufs[0].data = m_head;
        bufs[0].offset = 0;
        bufs[0].length = 0;

        data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

            // Copy data from demod_data to the mbuf
        rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                  ldpc_config.NumCbCodewLen());
        bufs[0].length += ldpc_config.NumCbCodewLen();

        rte_bbdev_op_data *bufs_out = *hard_outputs;
        m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[1]);

        bufs_out[0].data = m_head_out;
        bufs_out[0].offset = 0;
        bufs_out[0].length = 0;

        // Prepare the mbuf to receive the output data
        // char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
        // assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

        // BUG: This line causes a irregular stop of the program when fft_size = 4096,
        //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
        //      sampling rate.
        // rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());
        bufs_out[0].length += ldpc_config.NumCbCodewLen();

        ref_dec_op[1][enq_index_2]->ldpc_dec.input = *inputs[0];
        ref_dec_op[1][enq_index_2]->ldpc_dec.hard_output = *hard_outputs[0];

        rte_pktmbuf_free(m_head);
        rte_pktmbuf_free(m_head_out);

        if (kMubfLLRCheck){
          struct rte_mbuf *mbuf_1 = bufs[0].data;
          uint8_t *mbuf_data = rte_pktmbuf_mtod(mbuf_1, uint8_t *);
          size_t data_length = mbuf_1->data_len;
          
          if (kPrintMbufData) {
            size_t num_bytes = data_length;
            size_t num_32bit_values = num_bytes / 4;
            size_t remaining_bytes = num_bytes % 4;

            for (size_t k = 0; k < num_32bit_values; k++) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; j++) {
                value |= static_cast<uint32_t>(mbuf_data[k * 4 + j]) << (j * 8);
            }
            if (k > 0) {
                std::printf(", ");
            }
            std::printf("0x%08x", value);
            }

            // Handle remaining bytes if any
            if (remaining_bytes > 0) {
                if (num_32bit_values > 0) {
                    std::printf(", ");
                }
                uint32_t value = 0;
                for (size_t j = 0; j < remaining_bytes; j++) {
                    value |= static_cast<uint32_t>(mbuf_data[num_32bit_values * 4 + j]) << (j * 8);
                }
                std::printf("0x%08x", value);
            }

            std::printf("\n");
          }

          if (memcmp(mbuf_data, llr_buffer_ptr, data_length) == 0) {
            std::printf("Data in inputs[%u].data matches llr_buffer_ptr\n", 0 );
          } else {
            std::printf("Data mismatch in inputs[%zu].data\n", 0);

            // Print mismatched bytes
            for (size_t j = 0; j < data_length; ++j) {
                if (mbuf_data[j] != (uint8_t)llr_buffer_ptr[j]) {
                    std::printf("Mismatch at byte %zu: mbuf_data=%02x, llr_buffer_ptr=%02x\n",
                                j, mbuf_data[j], (uint8_t)llr_buffer_ptr[j]);
              }
            }
          }
        }
        enq_2 += rte_bbdev_enqueue_ldpc_dec_ops(1, 0, &ref_dec_op[1][enq_2], 1);

        size_t end_else = GetTime::WorkerRdtsc();
        size_t duration_else = end_else - start_tsc_else;

        duration_stat_->task_duration_[0] += duration_else;
        enq_index_2++;
    }
    else if (symbol_idx_ul % 4 == 2){ 
                  // std::cout<<"symbol_idx_ul is: " << symbol_idx_ul << ", ue_id is: " << ue_id << std::endl;
        size_t start_tsc_else = GetTime::WorkerRdtsc();

        int8_t *llr_buffer_ptr;
        struct rte_mbuf *m_head;
        struct rte_mbuf *m_head_out;

        llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                                (cfg_->ModOrderBits(Direction::kUplink) *
                                  (ldpc_config.NumCbCodewLen() * cur_cb_id));     

        if (symbol_idx_ul == 14 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
            std::printf("Hex: LLR data, symbol_offset, symbol_idx is: %zu\n", symbol_idx_ul);
            for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
                // Cast to uint8_t to ensure correct printing of the byte in hexadecimal
                std::printf("%02X ", static_cast<uint8_t>(*(llr_buffer_ptr + i)));
            }
            std::printf("\n");
        }
    
        char *data;
        struct rte_bbdev_op_data *bufs = *inputs;

        m_head = rte_pktmbuf_alloc(in_mbuf_pool[1]);

        bufs[0].data = m_head;
        bufs[0].offset = 0;
        bufs[0].length = 0;

        data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

            // Copy data from demod_data to the mbuf
        rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                  ldpc_config.NumCbCodewLen());
        bufs[0].length += ldpc_config.NumCbCodewLen();

        rte_bbdev_op_data *bufs_out = *hard_outputs;
        m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[1]);

        bufs_out[0].data = m_head_out;
        bufs_out[0].offset = 0;
        bufs_out[0].length = 0;

        // Prepare the mbuf to receive the output data
        // char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
        // assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

        // BUG: This line causes a irregular stop of the program when fft_size = 4096,
        //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
        //      sampling rate.
        // rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());
        bufs_out[0].length += ldpc_config.NumCbCodewLen();

        ref_dec_op[2][enq_index_3]->ldpc_dec.input = *inputs[0];
        ref_dec_op[2][enq_index_3]->ldpc_dec.hard_output = *hard_outputs[0];

        rte_pktmbuf_free(m_head);
        rte_pktmbuf_free(m_head_out);

        if (kMubfLLRCheck){
          struct rte_mbuf *mbuf_1 = bufs[0].data;
          uint8_t *mbuf_data = rte_pktmbuf_mtod(mbuf_1, uint8_t *);
          size_t data_length = mbuf_1->data_len;
          
          if (kPrintMbufData) {
            size_t num_bytes = data_length;
            size_t num_32bit_values = num_bytes / 4;
            size_t remaining_bytes = num_bytes % 4;

            for (size_t k = 0; k < num_32bit_values; k++) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; j++) {
                value |= static_cast<uint32_t>(mbuf_data[k * 4 + j]) << (j * 8);
            }
            if (k > 0) {
                std::printf(", ");
            }
            std::printf("0x%08x", value);
            }

            // Handle remaining bytes if any
            if (remaining_bytes > 0) {
                if (num_32bit_values > 0) {
                    std::printf(", ");
                }
                uint32_t value = 0;
                for (size_t j = 0; j < remaining_bytes; j++) {
                    value |= static_cast<uint32_t>(mbuf_data[num_32bit_values * 4 + j]) << (j * 8);
                }
                std::printf("0x%08x", value);
            }

            std::printf("\n");
          }

          if (memcmp(mbuf_data, llr_buffer_ptr, data_length) == 0) {
            std::printf("Data in inputs[%u].data matches llr_buffer_ptr\n", 0 );
          } else {
            std::printf("Data mismatch in inputs[%zu].data\n", 0);

            // Print mismatched bytes
            for (size_t j = 0; j < data_length; ++j) {
                if (mbuf_data[j] != (uint8_t)llr_buffer_ptr[j]) {
                    std::printf("Mismatch at byte %zu: mbuf_data=%02x, llr_buffer_ptr=%02x\n",
                                j, mbuf_data[j], (uint8_t)llr_buffer_ptr[j]);
              }
            }
          }
        }
        enq_3 += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[2], 0, &ref_dec_op[2][enq_3], 1);

        size_t end_else = GetTime::WorkerRdtsc();
        size_t duration_else = end_else - start_tsc_else;

        duration_stat_->task_duration_[0] += duration_else;
        enq_index_3++;
    }
    else {
        size_t start_tsc_else = GetTime::WorkerRdtsc();

        int8_t *llr_buffer_ptr;
        struct rte_mbuf *m_head;
        struct rte_mbuf *m_head_out;

        llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                                (cfg_->ModOrderBits(Direction::kUplink) *
                                  (ldpc_config.NumCbCodewLen() * cur_cb_id));     

        if (symbol_idx_ul == 14 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
            std::printf("Hex: LLR data, symbol_offset, symbol_idx is: %zu\n", symbol_idx_ul);
            for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
                // Cast to uint8_t to ensure correct printing of the byte in hexadecimal
                std::printf("%02X ", static_cast<uint8_t>(*(llr_buffer_ptr + i)));
            }
            std::printf("\n");
        }
    
        char *data;
        struct rte_bbdev_op_data *bufs = *inputs;

        m_head = rte_pktmbuf_alloc(in_mbuf_pool[1]);

        bufs[0].data = m_head;
        bufs[0].offset = 0;
        bufs[0].length = 0;

        data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

            // Copy data from demod_data to the mbuf
        rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                  ldpc_config.NumCbCodewLen());
        bufs[0].length += ldpc_config.NumCbCodewLen();

        rte_bbdev_op_data *bufs_out = *hard_outputs;
        m_head_out = rte_pktmbuf_alloc(out_mbuf_pool[1]);

        bufs_out[0].data = m_head_out;
        bufs_out[0].offset = 0;
        bufs_out[0].length = 0;

        // Prepare the mbuf to receive the output data
        // char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
        // assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

        // BUG: This line causes a irregular stop of the program when fft_size = 4096,
        //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
        //      sampling rate.
        // rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());
        bufs_out[0].length += ldpc_config.NumCbCodewLen();

        ref_dec_op[3][enq_index_4]->ldpc_dec.input = *inputs[0];
        ref_dec_op[3][enq_index_4]->ldpc_dec.hard_output = *hard_outputs[0];

        rte_pktmbuf_free(m_head);
        rte_pktmbuf_free(m_head_out);

        if (kMubfLLRCheck){
          struct rte_mbuf *mbuf_1 = bufs[0].data;
          uint8_t *mbuf_data = rte_pktmbuf_mtod(mbuf_1, uint8_t *);
          size_t data_length = mbuf_1->data_len;
          
          if (kPrintMbufData) {
            size_t num_bytes = data_length;
            size_t num_32bit_values = num_bytes / 4;
            size_t remaining_bytes = num_bytes % 4;

            for (size_t k = 0; k < num_32bit_values; k++) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; j++) {
                value |= static_cast<uint32_t>(mbuf_data[k * 4 + j]) << (j * 8);
            }
            if (k > 0) {
                std::printf(", ");
            }
            std::printf("0x%08x", value);
            }

            // Handle remaining bytes if any
            if (remaining_bytes > 0) {
                if (num_32bit_values > 0) {
                    std::printf(", ");
                }
                uint32_t value = 0;
                for (size_t j = 0; j < remaining_bytes; j++) {
                    value |= static_cast<uint32_t>(mbuf_data[num_32bit_values * 4 + j]) << (j * 8);
                }
                std::printf("0x%08x", value);
            }

            std::printf("\n");
          }

          if (memcmp(mbuf_data, llr_buffer_ptr, data_length) == 0) {
            std::printf("Data in inputs[%u].data matches llr_buffer_ptr\n", 0 );
          } else {
            std::printf("Data mismatch in inputs[%zu].data\n", 0);

            // Print mismatched bytes
            for (size_t j = 0; j < data_length; ++j) {
                if (mbuf_data[j] != (uint8_t)llr_buffer_ptr[j]) {
                    std::printf("Mismatch at byte %zu: mbuf_data=%02x, llr_buffer_ptr=%02x\n",
                                j, mbuf_data[j], (uint8_t)llr_buffer_ptr[j]);
              }
            }
          }
        }
        enq_4 += rte_bbdev_enqueue_ldpc_dec_ops(dev_ids[3], 0, &ref_dec_op[3][enq_4], 1);

        size_t end_else = GetTime::WorkerRdtsc();
        size_t duration_else = end_else - start_tsc_else;

        duration_stat_->task_duration_[0] += duration_else;
        enq_index_4++;
    }
  }
  return EventData(EventType::kDecode, tag);
}







