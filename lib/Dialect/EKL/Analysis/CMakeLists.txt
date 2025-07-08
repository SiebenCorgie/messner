add_mlir_library(EKLAnalysis
        Casting.cpp
        Extents.cpp
        Number.cpp
        TypeCheckingAdaptor.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen
)
