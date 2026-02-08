/**
 * @file agora_worker.h
 * @brief Declaration file for the main Agora worker class
 */

#ifndef AGORA_WORKERS2_H_
#define AGORA_WORKERS2_H_

#include <memory>
#include <thread>
#include <vector>

// #include "agora_buffer.h"
// #include "agora_bufferBase.h"
#include "agora_buffer_multiCell-S2.h"
#include "agora_workerBase.h"
#include "config.h"
#include "csv_logger.h"
#include "doer.h"
#include "mac_scheduler.h"
#include "mat_logger.h"
#include "phy_stats.h"
#include "stats.h"

class AgoraWorkerS2 : public WorkerBase {
 public:
  explicit AgoraWorkerS2(Config* cfg, MacScheduler* mac_sched, Stats* stats,
                         PhyStats* phy_stats, MessageInfoS2* message,
                         BufferBase* buffer, FrameInfoS2* frame);
  ~AgoraWorkerS2();

  void RunWorker() override {}

 private:
  void WorkerThread(int tid);
  void CreateThreads();
  void JoinThreads();

  std::vector<std::thread> workers_;

  const size_t base_worker_core_offset_;

  Config* const config_;

  MacScheduler* mac_sched_;
  Stats* stats_;
  PhyStats* phy_stats_;
  MessageInfoS2* message_;
  BufferBase* buffer_;
  FrameInfoS2* frame_;
};

#endif  // AGORA_WORKERS2_H_