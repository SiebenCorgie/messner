// RUN: messner-opt %s | FileCheck %s

ekl.program {
    //Broken for type-reasons
    ekl.func @brokenTypeInShape(%a: !ekl.array<f64[4, 4]>, %b: !ekl.array<ui32[4, 4]>) -> !ekl.array<f64[4, 4]>{
        %0 = "ekl.min"(%a, %b) : (!ekl.array<f64[4, 4]>, !ekl.array<ui32[4, 4]>) -> !ekl.array<f64[4, 4]>
        yield %0: !ekl.array<f64[4, 4]>
    }
}

