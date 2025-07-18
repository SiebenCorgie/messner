// RUN: messner-opt %s | FileCheck %s

ekl.program {
    //Broken for shape reason
    ekl.func @brokenShape(%a: !ekl.array<f64[3, 4]>, %b: !ekl.array<f64[4, 4]>) -> !ekl.array<f64[4, 4]>{
        %0 = "ekl.min"(%a, %b) : (!ekl.array<f64[3, 4]>, !ekl.array<f64[4, 4]>) -> !ekl.array<f64[4, 4]>
        yield %0 : !ekl.array<f64[4, 4]>
    }
}

