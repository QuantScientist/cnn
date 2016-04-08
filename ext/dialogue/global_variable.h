#pragma once
#include "cnn/cnn.h"
#include "cnn/data-util.h"

extern unsigned LAYERS;
extern unsigned HIDDEN_DIM;  // 1024
extern unsigned ALIGN_DIM;  // 1024
extern unsigned VOCAB_SIZE_SRC;
extern unsigned VOCAB_SIZE_TGT;
extern long nparallel;
extern long mbsize;
extern size_t g_train_on_turns; 

extern cnn::Dict sd;
extern cnn::Dict td;
extern cnn::stId2String<string> id2str;

extern int kSRC_SOS;
extern int kSRC_EOS;
extern int kTGT_SOS;
extern int kTGT_EOS;
extern int verbose;
extern int beam_search_decode;
extern cnn::real lambda;
extern int repnumber;

extern Sentence prv_response;

extern NumTurn2DialogId training_numturn2did;
extern NumTurn2DialogId devel_numturn2did;
extern NumTurn2DialogId test_numturn2did;
