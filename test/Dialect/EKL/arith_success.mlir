// RUN: messner-opt %s | FileCheck %s

ekl.program {
    
    ekl.func @myfunc(%a: !ekl.array<si32[4, 4]>, %b: !ekl.array<si32[4, 4]>) -> !ekl.array<si64[4, 4]>{
        %0 = "ekl.min"(%a, %b) : (!ekl.array<si32[4, 4]>, !ekl.array<si32[4, 4]>) -> !ekl.array<si32[4, 4]>
        %1 = ekl.promote %a: !ekl.array<si32[4, 4]> -> !ekl.array<si64[4, 4]>
        yield %1: !ekl.array<si64[4, 4]>
    }
}

