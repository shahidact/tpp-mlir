#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tune/TuneDialect.h"
#include "TPP/Dialect/Tune/TuneTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"

namespace nb = nanobind;

NB_MODULE(_tppDialects, m) {
  auto checkModule = m.def_submodule("check");

  checkModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        mlir::DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<mlir::check::CheckDialect, mlir::perf::PerfDialect,
                         mlir::xsmm::XsmmDialect>();
      },
      "registry");

  auto perfModule = m.def_submodule("perf");

  perfModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        mlir::DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<mlir::perf::PerfDialect>();
      },
      "registry");

  auto xsmmModule = m.def_submodule("xsmm");

  xsmmModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        mlir::DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<mlir::xsmm::XsmmDialect>();
      },
      "registry");

  auto tuneModule = m.def_submodule("tune");

  tuneModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        mlir::DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<mlir::tune::TuneDialect>();
      },
      "registry");

  auto transformModule = m.def_submodule("transform");
  auto transformTuneModule = transformModule.def_submodule("tune");

  transformTuneModule.def(
      "register_dialect_extension",
      [](MlirDialectRegistry wrappedRegistry) {
        mlir::DialectRegistry *registry = unwrap(wrappedRegistry);
        mlir::tune::registerTransformDialectExtension(*registry);
      },
      "registry");

  mlir::tpp::registerTppCompilerPasses();
  mlir::tpp::registerTppPassBundlePasses();
}
