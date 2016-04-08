#pragma once
#include "cnn/cnn.h"
#include "cnn/data-util.h"

using namespace cnn;

unsigned LAYERS = 2;
unsigned HIDDEN_DIM = 50;  // 1024
unsigned ALIGN_DIM = 25;  // 1024
unsigned VOCAB_SIZE_SRC = 0;
unsigned VOCAB_SIZE_TGT = 0;
long nparallel = -1;
long mbsize = -1;
size_t g_train_on_turns = 1;

cnn::Dict sd;
cnn::Dict td;
cnn::stId2String<string> id2str;

int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
int verbose;
int beam_search_decode;
cnn::real lambda = 1e-6;
int repnumber;

Sentence prv_response;

NumTurn2DialogId training_numturn2did;
NumTurn2DialogId devel_numturn2did;
NumTurn2DialogId test_numturn2did;

void reset_smoothed_ppl(vector<cnn::real>& ppl_hist);
cnn::real smoothed_ppl(cnn::real curPPL, vector<cnn::real>& ppl_hist);