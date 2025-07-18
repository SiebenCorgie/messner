// RUN: messner-opt %s | FileCheck %s

ekl.program {
    //Broken for casting reasons
    ekl.func @brokenCast(%a: !ekl.array<f64[4, 4]>) -> !ekl.array<f32[4, 4]>{
        %1 = ekl.coerce %a: !ekl.array<f64[4, 4]> -> !ekl.array<f32[4, 4]>
        yield %1: !ekl.array<f32[4, 4]>
    }
}

