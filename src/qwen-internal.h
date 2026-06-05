#pragma once
// qwen-internal.h: exposes the qt_context definition so tightly-coupled
// tools (qwen-tts.cpp) can access the PipelineTTS and its internals.
// Not part of the public ABI — only tools that ship inside the repo
// should include this.

#include "backend.h"
#include "bpe.h"
#include "pipeline-tts.h"

struct qt_context {
    BackendPair  bp;
    PipelineTTS  pt;
    BPETokenizer tok;
};
