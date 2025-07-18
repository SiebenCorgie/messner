// RUN: messner-opt %s | FileCheck %s
ekl.program {

    ekl.func @myfunc(%a: !ekl.array<si32[4, 4]>) -> !ekl.array<si32[4, 1]>{
        %0 = ekl.bcast %a: !ekl.array<si32[4, 4]> to [4, 1] -> !ekl.array<si32[4, 1]>
        yield %0: !ekl.array<si32[4, 1]>
    }
}

