#ifndef AGORA_WORKERBASE_H_
#define AGORA_WORKERBASE_H_

class WorkerBase {
 public:
  virtual ~WorkerBase() = default;
  virtual void RunWorker() = 0;  // Ensure derived classes implement this
};

#endif  // AGORA_WORKERBASE_H_
