#ifndef _TRAIN_PROC_H
#define _TRAIN_PROC_H

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/dnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cnn-helper.h"
#include "ext/dialogue/attention_with_intention.h"
#include "cnn/data-util.h"
#include "cnn/grad-check.h"
#include "cnn/metric-util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_wiarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_woarchive.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

#ifdef CUDA
#define NBR_DEV_PARALLEL_UTTS 2
#else
#define NBR_DEV_PARALLEL_UTTS 10
#endif

#define LEVENSHTEIN_THRESHOLD 5

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

#define MAX_NBR_TRUNS 200000
struct TrainingScores{
public:
    long swords; /// source side number of words
    long twords; /// target side number of words
    int training_score_current_location;
    int training_score_buf_size;
    cnn::real *training_scores;

    cnn::real dloss; 

public:
    TrainingScores(int bufsize) : training_score_buf_size(bufsize) {
        training_scores = (cnn::real*) cnn_mm_malloc(training_score_buf_size * sizeof(cnn::real), CNN_ALIGN);
        training_score_current_location = 0;
        swords = 0;
        twords = 0;
    }
    ~TrainingScores()
    {
        cnn_mm_free(training_scores);
    }

    void reset()
    {
        swords = 0;
        twords = 0;
        training_score_current_location = 0;
    }

    cnn::real compute_score()
    {
        if (training_score_current_location > MAX_NBR_TRUNS - 1)
            std::runtime_error("TrainingScore out of memory");

        dloss = 0;
        vector<cnn::real> raw_score = as_vector(training_score_current_location, training_scores); 
        for (auto& p : raw_score)
            dloss += p;
        return dloss;
    }
};

/**
The higher level training process
*/
template <class Proc>
class TrainProcess{
private:
    TrainingScores* training_set_scores;
    TrainingScores* dev_set_scores;

public:
    TrainProcess() {
        training_set_scores = new TrainingScores(MAX_NBR_TRUNS);
        dev_set_scores = new TrainingScores(MAX_NBR_TRUNS);
    }
    ~TrainProcess()
    {
        delete training_set_scores;
        delete dev_set_scores;
    }

    void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale);

    void batch_train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real& largest_cost, bool do_segmental_training, bool update_sgd, 
        bool doGradientCheck, bool b_inside_logic, 
        bool do_padding, int kEOS  /// do padding. if so, use kEOS as the padding symbol
        );
    void REINFORCEtrain(Model &model, Proc &am, Proc &am_agent_mirrow, Corpus &training, Corpus &devel, Trainer &sgd, string out_file, Dict & td, int max_epochs, int nparallel, cnn::real& largest_cost, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0);
    void split_data_batch_train(string train_filename, Model &model, Proc &am, Corpus &devel, Trainer &sgd, string out_file, int max_epochs, int nparallel, int epochsize, bool do_segmental_training, bool do_gradient_check, bool do_padding);
    
    /** report perplexity 

    @param words_s the word count in the source side
    @param words_t the word count in the target side

    @return entrpy loss
    */
    cnn::real testPPL(Model &model, Proc &am, Corpus &devel, NumTurn2DialogId& info, string out_file, bool segmental_training, cnn::real& words_s, cnn::real& words_t);
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo, bool segmental_training, const string& score_embedding_fn = "");
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test_segmental(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test(Model &model, Proc &am, TupleCorpus &devel, string out_file, Dict & sd, Dict & td);
    
    void dialogue(Model &model, Proc &am, string out_file, Dict & td);

    void collect_sample_responses(Proc& am, Corpus &training);

    void nosegmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt,
        TrainingScores* scores, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr);
    void segmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt, TrainingScores *scores, bool resetmodel, bool doGradientCheck = false, Trainer* sgd = nullptr);
    void REINFORCE_nosegmental_forward_backward(Model &model, Proc &am, Proc &am_mirrow, PDialogue &v_v_dialogues, int nutt,
        cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, Trainer* sgd, Dict& sd, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0,
        bool update_model = true);

public:
    vector<cnn::real> ppl_hist;

};

/**
this is fake experiment as the user side is known and supposedly respond correctly to the agent side
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo,
    bool segmental_training, const string& score_embedding_fn)
{
    unsigned lines = 0;

    ofstream of(out_file);

    Timer iteration("completed in");

    /// report BLEU score
    test(model, am, devel, out_file + "bleu", sd);

    dev_set_scores->reset(); 

    {
        vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
        size_t id_stt_diag_id = 0;
        PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
        vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
        size_t ndutt = id_sel_idx.size();

        if (verbose)
        {
            cerr << "selected " << ndutt << " :  ";
            for (auto p : id_sel_idx)
                cerr << p << " ";
            cerr << endl;
        }

        while (ndutt > 0)
        {
            if (segmental_training)
                segmental_forward_backward(model, am, vd_dialogues, ndutt, dev_set_scores, false);
            else
                nosegmental_forward_backward(model, am, vd_dialogues, ndutt, dev_set_scores, true);

            id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
            ndutt = id_sel_idx.size();

            if (verbose)
            {
                cerr << "selected " << ndutt << " :  ";
                for (auto p : id_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }
        }
    }

    dev_set_scores->compute_score();
    cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] E = " << (dev_set_scores->dloss / dev_set_scores->twords) << " ppl=" << exp(dev_set_scores->dloss / dev_set_scores->twords) << ' ';
    of << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] E = " << (dev_set_scores->dloss / dev_set_scores->twords) << " ppl=" << exp(dev_set_scores->dloss / dev_set_scores->twords) << ' ';

    of.close();
}

/**
Test perplexity on the corpus
*/
template <class AM_t>
cnn::real TrainProcess<AM_t>::testPPL(Model &model, AM_t &am, Corpus &devel, NumTurn2DialogId&  test_corpusinfo, string out_file, bool segmental_training, cnn::real & ddchars_s, cnn::real& ddchars_t)
{
    unsigned lines = 0;

    unsigned si = devel.size(); /// number of dialgoues in training
    if (si == 0)
        return LZERO;

    ofstream of(out_file, ios::app);

    Timer iteration("completed in");

    dev_set_scores->reset();

    vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
    size_t id_stt_diag_id = 0;
    PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
    vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);

    size_t ndutt = id_sel_idx.size();

    while (ndutt > 0)
    {
        if (segmental_training)
            segmental_forward_backward(model, am, vd_dialogues, ndutt, dev_set_scores, false);
        else
            nosegmental_forward_backward(model, am, vd_dialogues, ndutt, dev_set_scores, true);

        id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
        ndutt = id_sel_idx.size();

    }
    dev_set_scores->compute_score();
    cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] E = " << (dev_set_scores->dloss / dev_set_scores->twords) << " ppl=" << exp(dev_set_scores->dloss / dev_set_scores->twords) << ' ';

    of.close();

    return dev_set_scores->dloss;
}

/** warning, the test function use the true past response as the context, when measure bleu score
• So the BLEU score is artificially high
• However, because the use input is conditioned on the past response. If using the true decoder response as the past context, the user input cannot be from the corpus.
• Therefore, it is reasonable to use the true past response as context when evaluating the model.
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    BleuMetric bleuScore;
    bleuScore.Initialize();

    ofstream of(out_file);

    Timer iteration("completed in");

    for (auto diag : devel){

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            if (turn_id == 0)
                res = am.decode(turn.first, cg, sd);
            else
                res = am.decode(prv_turn.second, turn.first, cg, sd);

            if (turn.first.size() > 0)
            {
                cout << "source: ";
                for (auto p : turn.first){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (turn.second.size() > 0)
            {
                cout << "ref response: ";
                for (auto p : turn.second){
                    cout << sd.Convert(p) << " ";
                    sref.push_back(sd.Convert(p));
                }
                cout << endl;
            }

            if (res.size() > 0)
            {
                cout << "res response: ";
                for (auto p : res){
                    cout << sd.Convert(p) << " ";
                    srec.push_back(sd.Convert(p));
                }
                cout << endl;
            }


            bleuScore.AccumulateScore(sref, srec);

            turn_id++;
            prv_turn = turn;
        }
    }

    string sBleuScore = bleuScore.GetScore();
    cout << "BLEU (4) score = " << sBleuScore << endl;
    of << sBleuScore << endl;

    of.close();
}

/** warning, the test function use the true past response as the context, when measure bleu score
• So the BLEU score is artificially high
• However, because the use input is conditioned on the past response. If using the true decoder response as the past context, the user input cannot be from the corpus.
• Therefore, it is reasonable to use the true past response as context when evaluating the model.
*/
template <class AM_t>
void TrainProcess<AM_t>::test_segmental(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    unsigned lines = 0;
    cnn::real dloss = 0;
    cnn::real dchars_s = 0;
    cnn::real dchars_t = 0;

    BleuMetric bleuScore;
    bleuScore.Initialize();

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");
    cnn::real ddloss = 0;
    cnn::real ddchars_s = 0;
    cnn::real ddchars_t = 0;


    for (auto diag : devel){

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            if (turn_id == 0)
                res = am.decode(turn.first, cg, sd);
            else
                res = am.decode(prv_turn.second, turn.first, cg, sd);

            if (turn.first.size() > 0)
            {
                cout << "source: ";
                for (auto p : turn.first){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (turn.second.size() > 0)
            {
                cout << "ref response: ";
                for (auto p : turn.second){
                    cout << sd.Convert(p) << " ";
                    sref.push_back(sd.Convert(p));
                }
                cout << endl;
            }

            if (res.size() > 0)
            {
                cout << "res response: ";
                for (auto p : res){
                    cout << sd.Convert(p) << " ";
                    srec.push_back(sd.Convert(p));
                }
                cout << endl;
            }


            bleuScore.AccumulateScore(sref, srec);

            turn_id++;
            prv_turn = turn;
        }
    }

    string sBleuScore = bleuScore.GetScore();
    cout << "BLEU (4) score = " << sBleuScore << endl;
    of << sBleuScore << endl;

    of.close();
}

/**
Test on the tuple corpus 
output recognition results for each test
not using perplexity to report progresses
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, TupleCorpus &devel, string out_file, Dict & sd, Dict & td)
{
    unsigned lines = 0;
    cnn::real dloss = 0;
    cnn::real dchars_s = 0;
    cnn::real dchars_t = 0;

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");
    cnn::real ddloss = 0;
    cnn::real ddchars_s = 0;
    cnn::real ddchars_t = 0;

    for (auto diag : devel){

        SentenceTuple prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        ComputationGraph cg;
        vector<int> res;
        for (auto spair : diag){

            SentenceTuple turn = spair;

            if (turn_id == 0)
                res = am.decode_tuple(turn, cg, sd, td);
            else
                res = am.decode_tuple(prv_turn, turn, cg, sd, td);

            if (turn.first.size() > 0)
            {
                for (auto p : turn.first){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (turn.last.size() > 0)
            {
                for (auto p : turn.last){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (res.size() > 0)
            {
                for (auto p : res){
                    cout << td.Convert(p) << " ";
                }
                cout << endl;
            }

            turn_id++;
            prv_turn = turn;
        } 
    }


    of.close();
}

template <class AM_t>
void TrainProcess<AM_t>::dialogue(Model &model, AM_t &am, string out_file, Dict & td)
{
    string shuman;
    ofstream of(out_file);

    int d_idx = 0;
    while (1){
        cout << "please start dialogue with the agent. you can end this dialogue by typing exit " << endl;

        size_t t_idx = 0;
        vector<int> decode_output;
        vector<int> shuman_input;
        Sentence prv_response;
        ComputationGraph cg;
        while (1){
#ifdef INPUT_UTF8
            std::getline(wcin, shuman);
            if (shuman.find(L"exit") == 0)
                break;
#else
            std::getline(cin, shuman);
            if (shuman.find("exit") == 0)
                break;
#endif
            shuman = "<s> " + shuman + "</s>";
            convertHumanQuery(shuman, shuman_input, td);

            if (t_idx == 0)
                decode_output = am.decode(shuman_input, cg, td);
            else
                decode_output = am.decode(prv_response, shuman_input, cg, td);

            of << "res ||| " << d_idx << " ||| " << t_idx << " ||| ";
            for (auto pp : shuman_input)
            {
                of << td.Convert(pp) << " ";
            }
            of << " ||| ";

            for (auto pp : decode_output)
            {
                of << td.Convert(pp) << " ";
            }
            of << endl;

            cout << "Agent: ";
            for (auto pp : decode_output)
            {
                cout << td.Convert(pp) << " ";
            }
            cout << endl;

            prv_response = decode_output;
            t_idx++;
        }
        d_idx++;
        of << endl;
    }

    of.close();
}

/**
inspired by the following two papers
Sequence level training with recurrent neural networks http://arxiv.org/pdf/1511.06732v3.pdf
Minimum risk training for neural machine translation http://arxiv.org/abs/1512.02433

use decoded responses as targets. start this process from the last turn, and then gradually move to earlier turns.
this is also for implementation convenience.

/// initially alwasy use the xent, later on, with probability p, use the decoded response as target, but weight it
/// with a reward from BLEU
/// this probability is increased from 0 to 1.0.
/// two avoid different scaling, should apply decoding to all incoming sentences or otherwise, all use xent training

/// with probability p, decode an input
vector<int> response = s2tmodel_sim.decode(insent, cg);
/// evaluate the response to get BLEU score

/// subtract the BLEU score with a baseline number

/// the scalar is the reward signal

/// the target responses: some utterances are with true responses and the others are with decoded responses
*/
template <class AM_t>
void TrainProcess<AM_t>::REINFORCE_nosegmental_forward_backward(Model &model, AM_t &am, AM_t &am_mirrow, PDialogue &v_v_dialogues, int nutt,
    cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, Trainer* sgd, Dict& sd, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling, bool update_model)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    PTurn prv_turn, new_turn, new_prv_turn;
    BleuMetric bleuScore;
    bleuScore.Initialize();

    bool do_sampling = false; 
    cnn::real rng_value = rand() / (RAND_MAX + 0.0); 
    if (rng_value >= threshold_prob_for_sampling)
    {
        do_sampling = true; 
    }

    ComputationGraph cg;

    am.reset();
    am_mirrow.reset();

    /// train on two segments of a dialogue
    vector<Sentence> res;
    vector<Expression> v_errs; /// the errors to be minimized
    vector<cnn::real> v_bleu_score;
    vector<Expression> i_err;

    for (auto &turn : v_v_dialogues)
    {
        if (do_sampling)
        {
            vector<string> sref, srec;
            vector<Sentence> v_input, v_prv_response;

            v_bleu_score.clear();

            for (auto& p : turn)
            {
                v_input.push_back(p.first);
            }
            for (auto&p : prv_turn)
            {
                v_prv_response.push_back(p.second);
            }

            if (turn_id == 0)
            {
                res = am_mirrow.batch_decode(v_input, cg, sd);
            }
            else
            {
                res = am_mirrow.batch_decode(v_prv_response, v_input, cg, sd);
            }

            size_t k = 0;
            for (auto &q : res)
            {

                sref.clear();
                if (verbose) cout << "ref response: ";
                for (auto p : turn[k].second){
                    if (verbose) cout << sd.Convert(p) << " ";
                    sref.push_back(sd.Convert(p));
                }
                if (verbose) cout << endl;

                srec.clear();
                if (verbose) cout << "res response: ";
                for (auto p : q){
                    if (verbose) cout << sd.Convert(p) << " ";
                    srec.push_back(sd.Convert(p));
                }
                if (verbose) cout << endl;

                cnn::real score = bleuScore.GetSentenceScore(sref, srec);
                v_bleu_score.push_back(score);

                k++;
            }

            new_turn = turn;
            for (size_t k = 0; k < nutt; k++)
            {
                new_turn[k].second = res[k];
            }

            /// get errors from the decoded results
            if (turn_id == 0)
            {
                i_err = am.build_graph(new_turn, cg);
            }
            else
            {
                i_err = am.build_graph(new_prv_turn, new_turn, cg);
            }
        }
        else{
            /// get errors from the true reference
            if (turn_id == 0)
            {
                i_err = am.build_graph(turn, cg);
            }
            else
            {
                i_err = am.build_graph(prv_turn, turn, cg);
            }
        }

        if (do_sampling)
        {
            for (size_t k = 0; k < nutt; k++)
            {
                Expression t_err = i_err[k];
                v_errs.push_back(t_err * (v_bleu_score[k] - reward_baseline));  /// multiply with reward
            }
        }
        else
        {
            for (auto &p : i_err)
                v_errs.push_back(p);
        }

        cg.incremental_forward();

        prv_turn = turn;
        new_prv_turn = new_turn;
        turn_id++;
        i_turns++;
    }

    Expression i_total_err = sum(v_errs);
    dloss += as_scalar(cg.get_value(i_total_err));

    dchars_s += am.swords;
    dchars_t += am.twords;

    if (sgd != nullptr && update_model)
    {
        cg.backward();
        sgd->update(am.twords);
    }
}

template <class AM_t>
void TrainProcess<AM_t>::nosegmental_forward_backward(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt, TrainingScores* scores, bool resetmodel, int init_turn_id, Trainer* sgd)
{
    size_t turn_id = init_turn_id;
    size_t i_turns = 0;
    PTurn prv_turn;

    ComputationGraph cg;
    if (resetmodel)
    {
        am.reset();
    }

    for (auto turn : v_v_dialogues)
    {
        if (turn_id == 0)
        {
            am.build_graph(turn, cg);
        }
        else
        {
            am.build_graph(prv_turn, turn, cg);
        }

        cg.incremental_forward();
        //            CheckGrad(model, cg);

        prv_turn = turn;
        turn_id++;
        i_turns++;
    }

    Tensor tv = cg.get_value(am.s2txent.i);
    TensorTools::PushElementsToMemory(scores->training_score_current_location,
        scores->training_score_buf_size,
        scores->training_scores, tv); 

    scores->swords += am.swords;
    scores->twords += am.twords;


    if (sgd != nullptr)
    {
        cg.backward();
        sgd->update(am.twords);
    }
}

template <class AM_t>
void TrainProcess<AM_t>::segmental_forward_backward(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    PTurn prv_turn;

    if (verbose)
        cout << "start segmental_forward_backward" << endl;

    for (auto turn : v_v_dialogues)
    {
        ComputationGraph cg;
        if (resetmodel)
        {
            am.reset();
        }

        if (turn_id == 0)
        {
            am.build_graph(turn, cg);
        }
        else
        {
            am.build_graph(prv_turn, turn, cg);
        }

        if (verbose) cout << "after graph build" << endl;

        if (doGradientCheck 
            && turn_id > 3 // do gradient check after burn-in
            )
            CheckGrad(model, cg);

        if (sgd != nullptr)
        {
            if (verbose)
                cout << " start backprop " << endl;
            cg.backward();
            if (verbose)
                cout << " done backprop " << endl;
            sgd->update(am.twords);
            if (verbose)
                cout << " done update" << endl;
        }

        Tensor tv = cg.get_value(am.s2txent.i);
        TensorTools::PushElementsToMemory(scores->training_score_current_location,
            scores->training_score_buf_size,
            scores->training_scores,
            tv);

        scores->swords += am.swords;
        scores->twords += am.twords;

        prv_turn = turn;
        turn_id++;
        i_turns++;
    }
}

/**
Train with REINFORCE algorithm
*/
template <class AM_t>
void TrainProcess<AM_t>::REINFORCEtrain(Model &model, AM_t &am, AM_t &am_agent_mirrow, Corpus &training, Corpus &devel, Trainer &sgd, string out_file, Dict & td, int max_epochs, int nparallel, cnn::real& largest_cost, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling)
{
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    threshold_prob_for_sampling = min<cnn::real>(1.0, max<cnn::real>(0.0, threshold_prob_for_sampling)); /// normalize to [0.0, 1.0]

    bool first = true;
    int report = 0;
    unsigned lines = 0;

    ofstream out(out_file, ofstream::out);
    boost::archive::text_oarchive oa(out);
    oa << model;
    out.close();

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;

        for (unsigned iter = 0; iter < report_every_i;) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                /// shuffle number of turns
                shuffle(training_numturn2did.vNumTurns.begin(), training_numturn2did.vNumTurns.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
                for (auto p : training_numturn2did.mapNumTurn2DialogId){
                    /// shuffle dailogues with the same number of turns
                    random_shuffle(p.second.begin(), p.second.end());
                }
                v_selected.assign(training.size(), false);
            }

            Dialogue prv_turn;
            
            PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
            vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
            size_t nutt = i_sel_idx.size();

            REINFORCE_nosegmental_forward_backward(model, am, am_agent_mirrow, v_dialogues, nutt, dloss, dchars_s, dchars_t, &sgd, td, reward_baseline, threshold_prob_for_sampling);
            si += nutt;
            lines += nutt;
            iter += nutt;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (cnn::real)training.size()) == 0.0) {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

            vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
            size_t id_stt_diag_id = 0;
            PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
            vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
            size_t ndutt = id_sel_idx.size();

            while (ndutt > 0)
            {
                /// the cost is -(r - r_baseline) * log P
                /// for small P, but with large r, the cost is high, so to reduce it, it generates large gradient as this event corresponds to low probability but high reward
                REINFORCE_nosegmental_forward_backward(model, am, am_agent_mirrow, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, nullptr, td, reward_baseline, 0.0, false);
                
                id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                ndutt = id_sel_idx.size();
            }
            ddloss = smoothed_ppl(ddloss, ppl_hist);
            if (ddloss < largest_cost) {
                largest_cost = ddloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] cost = " << (ddloss / ddchars_t) << " approximate ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}


/* the following does mutiple sentences per minibatch
@ b_inside_logic : use logic inside of batch to do evaluation on the dev set. if it is false, do dev set evaluation only if sgd.epoch changes
*/
template <class AM_t>
void TrainProcess<AM_t>::batch_train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
    bool sgd_update_epochs, bool do_gradient_check, bool b_inside_logic, 
    bool b_do_padding, int kEOS /// for padding if so use kEOS as the padding symbol
    )
{
    if (verbose)
        cout << "batch_train: "; 
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;

    if (b_inside_logic) 
        reset_smoothed_ppl(ppl_hist);

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    /// if no update of sgd in this function, need to train with all data in one pass and then return
    if (sgd_update_epochs == false)
    {
        report_every_i = training.size();
        si = 0;
    }

    while ((sgd_update_epochs && sgd.epoch < max_epochs) ||  /// run multiple passes of data
           (!sgd_update_epochs && si < training.size()))  /// run one pass of the data
    {
        Timer iteration("completed in");
        training_set_scores->reset();

        PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers

        for (unsigned iter = 0; iter < report_every_i;) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else if (sgd_update_epochs){
                    sgd.update_epoch();
                    lines -= training.size();
                }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                /// shuffle number of turns
                shuffle(training_numturn2did.vNumTurns.begin(), training_numturn2did.vNumTurns.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
                for (auto p : training_numturn2did.mapNumTurn2DialogId){
                    /// shuffle dailogues with the same number of turns
                    random_shuffle(p.second.begin(), p.second.end());
                }
                v_selected.assign(training.size(), false);
            }

            Dialogue prv_turn;
            vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
            size_t nutt = i_sel_idx.size();
            if (nutt == 0)
                break;

            if (b_do_padding)
            {
                /// padding all input and output in each turn into same length with </s> symbol
                /// padding </s> to the front for source side
                /// padding </s> to the back for target side
                PDialogue pd = padding_with_eos(v_dialogues, kEOS, { false, true});
                v_dialogues = pd;
            }

            if (verbose)
            {
                cerr << "selected " << nutt << " :  ";
                for (auto p : i_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }

            if (segmental_training)
                segmental_forward_backward(model, am, v_dialogues, nutt, training_set_scores, false, do_gradient_check, &sgd);
            else
                nosegmental_forward_backward(model, am, v_dialogues, nutt, training_set_scores, true, 0, &sgd);

            si += nutt;
            lines += nutt;
            iter += nutt;
        }

        training_set_scores->compute_score();

        sgd.status();
        iteration.WordsPerSecond(training_set_scores->twords + training_set_scores->swords);
        cerr << "\n***Train " << (lines / (cnn::real)training.size()) * 100 << " %100 of epoch[" << sgd.epoch << "] E = " << (training_set_scores->dloss / training_set_scores->twords) << " ppl=" << exp(training_set_scores->dloss / training_set_scores->twords) << ' ';


        vector<SentencePair> vs;
        for (auto&p : v_dialogues)
            vs.push_back(p[0]);
        vector<SentencePair> vres;
        am.respond(vs, vres, sd);
        
        // show score on dev data?
        report++;

        if (b_inside_logic && devel.size() > 0 && (floor(sgd.epoch) != prv_epoch
                                 || (report % dev_every_i_reports == 0
                                                        || fmod(lines, (cnn::real)training.size()) == 0.0))) 
        {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

            ddloss = testPPL(model, am, devel, devel_numturn2did, out_file + ".dev.log", segmental_training, ddchars_s, ddchars_t);

            ddloss = smoothed_ppl(ddloss, ppl_hist);
            if (ddloss < best) {
                best = ddloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);

        if (sgd_update_epochs == false)
        {
            /// because there is no update on sgd epoch, this loop can run forever. 
            /// so just run one iteration and quit
            break;
        }
        else{
            ofstream out(out_file + "e" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
        }
    }
}

/**
collect sample responses
*/
template <class AM_t>
void TrainProcess<AM_t>::collect_sample_responses(AM_t& am, Corpus &training)
{
    am.clear_candidates();
    for (auto & ds: training){
        vector<SentencePair> prv_turn;

        for (auto& spair : ds){
            SentencePair turn = spair;
            am.collect_candidates(spair.second);
        }
    }
}

/** 
since the tool loads data into memory and that can cause memory exhaustion, this function do sampling of data for each epoch.
*/
template <class AM_t>
void TrainProcess<AM_t>::split_data_batch_train(string train_filename, Model &model, AM_t &am, Corpus &devel, 
    Trainer &sgd, string out_file, 
    int max_epochs, int nparallel, int epochsize, bool segmental_training,
    bool do_gradient_check, bool do_padding)
{
    cnn::real largest_cost = 9e+99;
    cnn::real largest_dev_cost = 9e+99;

    reset_smoothed_ppl(ppl_hist);

    DataReader dr(train_filename);
    int trial = 0;
    dr.start(sd, kSRC_SOS, kSRC_EOS, epochsize);
    dr.join();
    
    Corpus training = dr.corpus();
    training_numturn2did = get_numturn2dialid(training);

    while (sgd.epoch < max_epochs)
    {
        Timer this_epoch("this epoch completed in");

        dr.detach();
        dr.start(sd, kSRC_SOS, kSRC_EOS, epochsize);

        batch_train(model, am, training, devel, sgd, out_file, 1, nparallel, largest_cost, segmental_training, false, do_gradient_check, false, do_padding, kSRC_EOS);

        dr.join(); /// synchroze data thread and main thread
        training = dr.corpus();
        training_numturn2did = get_numturn2dialid(training);

        if (training.size() == 0)
        {
            dr.restart(); 
            dr.start(sd, kSRC_SOS, kSRC_EOS, epochsize);
            dr.join(); /// make sure data is completely read
            training = dr.corpus();  /// copy the data from data thread to the data to be used in the main thread
            training_numturn2did = get_numturn2dialid(training);
//#define DEBUG
#ifndef DEBUG
            ofstream out(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
#endif
            sgd.update_epoch();

#ifndef DEBUG
            if (devel.size() > 0)
            {
                cnn::real ddloss, ddchars_s, ddchars_t;
                ddloss = testPPL(model, am, devel, devel_numturn2did, out_file + ".dev.log", segmental_training, ddchars_s, ddchars_t);

                ddloss = smoothed_ppl(ddloss, ppl_hist);
                if (ddloss < largest_dev_cost) {
                    /// save the model with the best performance on the dev set
                    largest_dev_cost = ddloss;

                    ofstream out(out_file, ofstream::out);
                    boost::archive::text_oarchive oa(out);
                    oa << model;
                    out.close();
                }
                else{
                    sgd.eta0 *= 0.5; /// reduce learning rate
                    sgd.eta *= 0.5; /// reduce learning rate
                }
            }
#endif
        }

        trial++;
    }
}

template <class Proc>
class ClassificationTrainProcess : public TrainProcess<Proc>{
public:
    ClassificationTrainProcess(){
    }

    void split_data_batch_train(string train_filename, Model &model, Proc &am, Corpus &devel, Trainer &sgd, string out_file, int max_epochs, int nparallel, int epochsize, bool do_segmental_training, bool do_gradient_check);
    
    void batch_train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
        bool sgd_update_epochs, bool do_gradient_check, bool b_inside_logic);

private:
    vector<cnn::real> ppl_hist;

};


/**
since the tool loads data into memory and that can cause memory exhaustion, this function do sampling of data for each epoch.
*/
template <class AM_t>
void ClassificationTrainProcess<AM_t>::split_data_batch_train(string train_filename, Model &model, AM_t &am, Corpus &devel,
    Trainer &sgd, string out_file,
    int max_epochs, int nparallel, int epochsize, bool segmental_training, bool do_gradient_check)
{
    // a mirrow of the agent to generate decoding results so that their results can be evaluated
    // this is not efficient implementation, better way is to share model parameters
    cnn::real largest_cost = 9e+99;

    ifstream ifs(train_filename);
    int trial = 0;
    while (sgd.epoch < max_epochs)
    {
        cerr << "Reading training data from " << train_filename << "...\n";
        Corpus training = read_corpus(ifs, sd, kSRC_SOS, kSRC_EOS, epochsize, make_pair<int,int>(2,4), make_pair<bool, bool>(true, false),
            id2str.phyId2logicId);
        training_numturn2did = get_numturn2dialid(training);

        if (ifs.eof() || training.size() == 0)
        {
            ifs.close();
            ifs.open(train_filename);

            if (training.size() == 0)
            {
                continue;
            }
            ofstream out(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();

            sgd.update_epoch();
        }

        batch_train(model, am, training, devel, sgd, out_file, 1, nparallel, largest_cost, segmental_training, false, do_gradient_check, false);

        if (fmod(trial, 50) == 0)
        {
            ofstream out(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
        }
        trial++;
    }
    ifs.close();
}

/* 
@ b_inside_logic : use logic inside of batch to do evaluation on the dev set. if it is false, do dev set evaluation only if sgd.epoch changes
*/
template <class AM_t>
void ClassificationTrainProcess<AM_t>::batch_train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
    bool sgd_update_epochs, bool doGradientCheck, bool b_inside_logic)
{
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    reset_smoothed_ppl(ppl_hist);

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    /// if no update of sgd in this function, need to train with all data in one pass and then return
    if (sgd_update_epochs == false)
    {
        report_every_i = training.size();
        si = 0;
    }

    while ((sgd_update_epochs && sgd.epoch < max_epochs) ||  /// run multiple passes of data
        (!sgd_update_epochs && si < training.size()))  /// run one pass of the data
    {
        Timer iteration("completed in");
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

        PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers

        for (unsigned iter = 0; iter < report_every_i;) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else if (sgd_update_epochs){
                    sgd.update_epoch();
                    lines -= training.size();
                }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                /// shuffle number of turns
                shuffle(training_numturn2did.vNumTurns.begin(), training_numturn2did.vNumTurns.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
                for (auto p : training_numturn2did.mapNumTurn2DialogId){
                    /// shuffle dailogues with the same number of turns
                    random_shuffle(p.second.begin(), p.second.end());
                }
                v_selected.assign(training.size(), false);
            }

            Dialogue prv_turn;
            size_t turn_id = 0;
            vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
            size_t nutt = i_sel_idx.size();
            if (nutt == 0)
                break;

            if (verbose)
            {
                cerr << "selected " << nutt << " :  ";
                for (auto p : i_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }

            if (segmental_training)
                segmental_forward_backward(model, am, v_dialogues, nutt, dloss, dchars_s, dchars_t, false, doGradientCheck, &sgd);
            else
                nosegmental_forward_backward(model, am, v_dialogues, nutt, dloss, dchars_s, dchars_t, true, 0, &sgd);

            si += nutt;
            lines += nutt;
            iter += nutt;
        }

        sgd.status();
        iteration.WordsPerSecond(dchars_t + dchars_s);
        cerr << "\n***Train " << (lines / (cnn::real)training.size()) * 100 << " %100 of epoch[" << sgd.epoch << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';


        vector<SentencePair> vs;
        for (auto&p : v_dialogues)
            vs.push_back(p[0]);
        vector<SentencePair> vres;
        am.respond(vs, vres, sd, id2str);

        // show score on dev data?
        report++;

        if (b_inside_logic && devel.size() > 0 && (floor(sgd.epoch) != prv_epoch
                                 || (report % dev_every_i_reports == 0 
                                                     || fmod(lines, (cnn::real)training.size()) == 0.0))) {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

            {
                vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
                size_t id_stt_diag_id = 0;
                PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
                vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                size_t ndutt = id_sel_idx.size();

                if (verbose)
                {
                    cerr << "selected " << ndutt << " :  ";
                    for (auto p : id_sel_idx)
                        cerr << p << " ";
                    cerr << endl;
                }

                while (ndutt > 0)
                {
                    if (segmental_training)
                        segmental_forward_backward(model, am, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, false);
                    else
                        nosegmental_forward_backward(model, am, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, true);

                    id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                    ndutt = id_sel_idx.size();

                    if (verbose)
                    {
                        cerr << "selected " << ndutt << " :  ";
                        for (auto p : id_sel_idx)
                            cerr << p << " ";
                        cerr << endl;
                    }
                }
            }
            ddloss = smoothed_ppl(ddloss, ppl_hist);
            if (ddloss < best) {
                best = ddloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);

        if (sgd_update_epochs == false)
        {
            /// because there is no update on sgd epoch, this loop can run forever. 
            /// so just run one iteration and quit
            break;
        }
        else{
            ofstream out(out_file + "e" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
        }
    }
}


#endif
