/**
 * @file agora_buffer.h
 * @brief Declaration file for the AgoraBuffer class
 */

#ifndef AGORA_BUFFERS2_H_
#define AGORA_BUFFERS2_H_

#include <array>
#include <cstddef>
#include <queue>

#include "agora_bufferBase.h"
#include "common_typedef_sdk.h"
#include "concurrent_queue_wrapper.h"
#include "concurrentqueue.h"
#include "config.h"
#include "mac_scheduler.h"
#include "memory_manage.h"
#include "message.h"
#include "symbols.h"
#include "utils.h"

class AgoraBufferS2 : public BufferBase {
 public:
  explicit AgoraBufferS2(Config* const cfg);
  // Delete copy constructor and copy assignment
  AgoraBufferS2(AgoraBufferS2 const&) = delete;
  AgoraBufferS2& operator=(AgoraBufferS2 const&) = delete;
  ~AgoraBufferS2();

  void AllocateTables() override;
  void FreeTables() override;
  void AllocatePhaseShifts();

  inline PtrGrid<kFrameWnd, kMaxUEs, complex_float>& GetCsi() override {
    return csi_buffer_;
  }
  inline PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetUlBeamMatrix()
      override {
    return ul_beam_matrix_;
  }
  inline PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetDlBeamMatrix()
      override {
    return dl_beam_matrix_;
  }
  inline PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDemod() override {
    return demod_buffer_;
  }

  inline PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDecod() override {
    return decoded_buffer_;
  }
  inline Table<complex_float>& GetFft() override { return fft_buffer_; }
  inline Table<complex_float>& GetEqual() override { return equal_buffer_; }
  inline Table<complex_float>& GetUeSpecPilot() override {
    return ue_spec_pilot_buffer_;
  }
  inline Table<complex_float>& GetIfft() override { return dl_ifft_buffer_; }
  inline Table<complex_float>& GetCalibUlMsum() override {
    return calib_ul_msum_buffer_;
  }
  inline Table<complex_float>& GetCalibDlMsum() override {
    return calib_dl_msum_buffer_;
  }
  inline Table<std::complex<int16_t>> GetDlBcastSignal() override {
    return dl_bcast_socket_buffer_;
  }
  inline Table<int8_t>& GetDlModBits() override { return dl_mod_bits_buffer_; }
  inline Table<int8_t>& GetDlBits() override { return dl_bits_buffer_; }
  inline Table<int8_t>& GetDlBitsStatus() override {
    return dl_bits_buffer_status_;
  }

  inline size_t GetUlSocketSize() const override { return ul_socket_buf_size_; }
  inline Table<char>& GetUlSocket() override { return ul_socket_buffer_; }
  inline char* GetDlSocket() override { return dl_socket_buffer_; }
  inline Table<complex_float>& GetCalibUl() override {
    return calib_ul_buffer_;
  }
  inline Table<complex_float>& GetCalibDl() override {
    return calib_dl_buffer_;
  }
  inline Table<complex_float>& GetCalib() override { return calib_buffer_; }

  inline std::array<arma::fmat, kFrameWnd>& GetUlPhaseBase() override {
    return ul_phase_base_;
  }
  inline std::array<arma::fmat, kFrameWnd>& GetUlPhaseShiftPerSymbol()
      override {
    return ul_phase_shift_per_symbol_;
  }

 private:
  // void AllocateTables();
  // void AllocatePhaseShifts();
  // void FreeTables();

  Config* const config_;
  const size_t ul_socket_buf_size_;

  PtrGrid<kFrameWnd, kMaxUEs, complex_float> csi_buffer_;
  PtrGrid<kFrameWnd, kMaxDataSCs, complex_float> ul_beam_matrix_;
  PtrGrid<kFrameWnd, kMaxDataSCs, complex_float> dl_beam_matrix_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> demod_buffer_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> decoded_buffer_;
  Table<complex_float> fft_buffer_;
  Table<complex_float> equal_buffer_;
  Table<complex_float> ue_spec_pilot_buffer_;
  Table<complex_float> dl_ifft_buffer_;
  Table<complex_float> calib_ul_msum_buffer_;
  Table<complex_float> calib_dl_msum_buffer_;
  Table<complex_float> calib_buffer_;
  Table<int8_t> dl_mod_bits_buffer_;
  Table<int8_t> dl_bits_buffer_;
  Table<int8_t> dl_bits_buffer_status_;
  Table<std::complex<int16_t>> dl_bcast_socket_buffer_;

  std::array<arma::fmat, kFrameWnd> ul_phase_base_;
  std::array<arma::fmat, kFrameWnd> ul_phase_shift_per_symbol_;

  Table<char> ul_socket_buffer_;
  char* dl_socket_buffer_;
  Table<complex_float> calib_ul_buffer_;
  Table<complex_float> calib_dl_buffer_;
};

struct SchedInfo {
  moodycamel::ConcurrentQueue<EventData> concurrent_q_;
  moodycamel::ProducerToken* ptok_;
};

// Used to communicate between the manager and the streamer/worker class
// Needs to manage its own memory
class MessageInfoS2 {
 public:
  explicit MessageInfoS2(size_t queue_size, size_t rx_queue_size,
                         size_t num_socket_thread)
      : num_socket_thread_S2(num_socket_thread) {
    tx_concurrent_queue_S2 = moodycamel::ConcurrentQueue<EventData>(queue_size);
    rx_concurrent_queue_S2 =
        moodycamel::ConcurrentQueue<EventData>(rx_queue_size);

    for (size_t i = 0; i < num_socket_thread; i++) {
      rx_ptoks_ptr_S2_[i] =
          new moodycamel::ProducerToken(rx_concurrent_queue_S2);
      tx_ptoks_ptr_S2_[i] =
          new moodycamel::ProducerToken(tx_concurrent_queue_S2);
    }

    // Allocate memory for the task concurrent queues
    Alloc(queue_size);
  }
  ~MessageInfoS2() {
    for (size_t i = 0; i < num_socket_thread_S2; i++) {
      delete rx_ptoks_ptr_S2_[i];
      delete tx_ptoks_ptr_S2_[i];
      rx_ptoks_ptr_S2_[i] = nullptr;
      tx_ptoks_ptr_S2_[i] = nullptr;
    }

    // Free memory for the task concurrent queues
    Free();
  }

  inline moodycamel::ConcurrentQueue<EventData>* GetTxConQ_S2() {
    return &tx_concurrent_queue_S2;
  }
  inline moodycamel::ConcurrentQueue<EventData>* GetRxConQ_S2() {
    return &rx_concurrent_queue_S2;
  }
  inline moodycamel::ProducerToken** GetTxPTokPtr_S2() {
    return tx_ptoks_ptr_S2_;
  }
  inline moodycamel::ProducerToken** GetRxPTokPtr_S2() {
    return rx_ptoks_ptr_S2_;
  }
  inline moodycamel::ProducerToken* GetTxPTokPtr_S2(size_t idx) {
    return tx_ptoks_ptr_S2_[idx];
  }
  inline moodycamel::ProducerToken* GetRxPTokPtr_S2(size_t idx) {
    return rx_ptoks_ptr_S2_[idx];
  }

  inline moodycamel::ProducerToken* GetPtok_S2(EventType event_type,
                                               size_t qid) {
    return task_queue_S2_.at(qid).at(static_cast<size_t>(event_type)).ptok_;
  }
  inline moodycamel::ConcurrentQueue<EventData>* GetTaskQueue_S2(
      EventType event_type, size_t qid) {
    return &task_queue_S2_.at(qid)
                .at(static_cast<size_t>(event_type))
                .concurrent_q_;
  }
  inline moodycamel::ConcurrentQueue<EventData>& GetCompQueue_S2(size_t qid) {
    return complete_task_queue_S2_.at(qid);
  }
  inline moodycamel::ProducerToken* GetWorkerPtok_S2(size_t qid,
                                                     size_t worker_id) {
    return worker_ptoks_ptr_S2_.at(qid).at(worker_id);
  }
  inline void EnqueueEventTaskQueue_S2(EventType event_type, size_t qid,
                                       EventData event) {
    TryEnqueueFallback(this->GetTaskQueue_S2(event_type, qid),
                       this->GetPtok_S2(event_type, qid), event);
  }
  inline size_t DequeueEventCompQueueBulk_S2(
      size_t qid, std::vector<EventData>& events_list) {
    return this->GetCompQueue_S2(qid).try_dequeue_bulk(&events_list.at(0),
                                                       events_list.size());
  }

 private:
  size_t num_socket_thread_S2;
  // keep the concurrent queue to communicate to streamer thread
  moodycamel::ConcurrentQueue<EventData> tx_concurrent_queue_S2;
  moodycamel::ConcurrentQueue<EventData> rx_concurrent_queue_S2;
  moodycamel::ProducerToken* rx_ptoks_ptr_S2_[kMaxThreads];
  moodycamel::ProducerToken* tx_ptoks_ptr_S2_[kMaxThreads];

  std::array<std::array<SchedInfo, kNumEventTypes>, kScheduleQueues>
      task_queue_S2_;
  std::array<moodycamel::ConcurrentQueue<EventData>, kScheduleQueues>
      complete_task_queue_S2_;
  std::array<std::array<moodycamel::ProducerToken*, kMaxThreads>,
             kScheduleQueues>
      worker_ptoks_ptr_S2_;

  inline void Alloc(size_t queue_size) {
    // Allocate memory for the task concurrent queues
    for (auto& queue : complete_task_queue_S2_) {
      queue = moodycamel::ConcurrentQueue<EventData>(queue_size);
    }
    for (auto& queue : task_queue_S2_) {
      for (auto& event : queue) {
        event.concurrent_q_ =
            moodycamel::ConcurrentQueue<EventData>(queue_size);
        event.ptok_ = new moodycamel::ProducerToken(event.concurrent_q_);
      }
    }

    size_t queue_count = 0;
    for (auto& queue : worker_ptoks_ptr_S2_) {
      for (auto& worker : queue) {
        worker = new moodycamel::ProducerToken(
            complete_task_queue_S2_.at(queue_count));
      }
      queue_count++;
    }
  }

  inline void Free() {
    for (auto& queue : task_queue_S2_) {
      for (auto& event : queue) {
        delete event.ptok_;
        event.ptok_ = nullptr;
      }
    }
    for (auto& queue : worker_ptoks_ptr_S2_) {
      for (auto& worker : queue) {
        delete worker;
        worker = nullptr;
      }
    }
  }
};

struct FrameInfoS2 {
  size_t cur_sche_frame_id_S2_;
  size_t cur_proc_frame_id_S2_;
};

#endif  // AGORA_BUFFERS2_H_
