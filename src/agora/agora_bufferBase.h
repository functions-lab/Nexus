#ifndef BUFFER_BASE_H_
#define BUFFER_BASE_H_

#include <array>
#include <cstddef>
#include <queue>

#include "common_typedef_sdk.h"
#include "concurrent_queue_wrapper.h"
#include "concurrentqueue.h"
#include "config.h"
#include "mac_scheduler.h"
#include "memory_manage.h"
#include "message.h"
#include "symbols.h"
#include "utils.h"

class BufferBase {
 public:
  virtual ~BufferBase() = default;

  // virtual void Initialize() = 0;  // Must be implemented by derived classes
  virtual void AllocateTables() = 0;
  virtual void FreeTables() = 0;

  // Add all required virtual getters
  virtual PtrGrid<kFrameWnd, kMaxUEs, complex_float>& GetCsi() = 0;
  virtual PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetUlBeamMatrix() = 0;
  virtual PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetDlBeamMatrix() = 0;
  virtual PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDemod() = 0;
  virtual PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDecod() = 0;
  virtual Table<complex_float>& GetFft() = 0;
  virtual Table<complex_float>& GetEqual() = 0;
  virtual Table<complex_float>& GetUeSpecPilot() = 0;
  virtual Table<complex_float>& GetIfft() = 0;
  virtual Table<complex_float>& GetCalibUlMsum() = 0;
  virtual Table<complex_float>& GetCalibDlMsum() = 0;
  virtual Table<std::complex<int16_t>> GetDlBcastSignal() = 0;
  virtual Table<int8_t>& GetDlModBits() = 0;
  virtual Table<int8_t>& GetDlBits() = 0;
  virtual Table<int8_t>& GetDlBitsStatus() = 0;
  virtual size_t GetUlSocketSize() const = 0;
  virtual Table<char>& GetUlSocket() = 0;
  virtual char* GetDlSocket() = 0;
  virtual Table<complex_float>& GetCalibUl() = 0;
  virtual Table<complex_float>& GetCalibDl() = 0;
  virtual Table<complex_float>& GetCalib() = 0;
  virtual std::array<arma::fmat, kFrameWnd>& GetUlPhaseBase() = 0;
  virtual std::array<arma::fmat, kFrameWnd>& GetUlPhaseShiftPerSymbol() = 0;
};

#endif  // BUFFER_BASE_H_