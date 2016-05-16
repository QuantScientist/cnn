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
#include "cnn/model.h"
#include "cnn/expr.h"
#include "cnn/cnn-helper.h"
#include "ext/dialogue/attention_with_intention.h"
#include "ext/lda/lda.h"
#include "ext/ngram/ngram.h"
#include "cnn/data-util.h"
#include "cnn/grad-check.h"
#include "cnn/metric-util.h"
#include "ext/trainer/eval_proc.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_wiarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_woarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
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

extern unsigned LAYERS;
extern unsigned HIDDEN_DIM;  // 1024
extern unsigned ALIGN_DIM;  // 1024
extern unsigned VOCAB_SIZE_SRC;
extern unsigned VOCAB_SIZE_TGT;
extern long nparallel;
extern long mbsize;
extern size_t g_train_on_turns;
extern cnn::real weight_IDF;

extern cnn::Dict sd;
extern cnn::Dict td;
extern cnn::stId2String<string> id2str;

extern int kSRC_SOS;
extern int kSRC_EOS;
extern int kTGT_SOS;
extern int kTGT_EOS;
extern int verbose;
extern int beam_search_decode;
extern cnn::real lambda; // = 1e-6;
extern int repnumber;
extern int rerankIDF;
extern int reinforceIDF;
extern cnn::real weight_IDF;
extern cnn::real weight_edist;

extern Sentence prv_response;

extern NumTurn2DialogId training_numturn2did;
extern NumTurn2DialogId devel_numturn2did;
extern NumTurn2DialogId test_numturn2did;

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

    TFIDFMetric * ptr_tfidfScore;;

public:
    TrainProcess() {
        training_set_scores = new TrainingScores(MAX_NBR_TRUNS);
        dev_set_scores = new TrainingScores(MAX_NBR_TRUNS);
        ptr_tfidfScore = nullptr;
    }
    ~TrainProcess()
    {
        delete training_set_scores;
        delete dev_set_scores;

        if (ptr_tfidfScore)
            delete ptr_tfidfScore;
    }

    void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale);

    void batch_train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real& largest_cost, bool do_segmental_training, bool update_sgd,
        bool doGradientCheck, bool b_inside_logic,
        bool do_padding, int kEOS,  /// do padding. if so, use kEOS as the padding symbol
        bool b_use_additional_feature
        );
    void supervised_pretrain(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, cnn::real target_ppl, int min_diag_id,
        bool bcharlevel, bool nosplitdialogue);

    void batch_train_ranking(Model &model, Proc &am, size_t max_epochs, Corpus &train_corpus, string model_out_fn, 
		string out_file, Dict & td, NumTurn2DialogId& train_corpusinfo, Trainer *sgd, int nparallel);

    /// adaptation using a small adaptation
    void online_adaptation(Model &model, Proc &am,
        const Dialogue & training, // user_input_target_response_pair,
        Trainer &sgd,
        const cnn::real& target_ppl, /// the target training ppl
        int maxepoch,                /// the maximum number of epochs
        const string & updated_model_fname);

    void train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs,
        bool bcharlevel, bool nosplitdialogue);
    void train(Model &model, Proc &am, TupleCorpus &training, Trainer &sgd, string out_file, int max_epochs);
    void split_data_batch_train(string train_filename, Model &model, Proc &am, Corpus &devel, Trainer &sgd, string out_file, int max_epochs, int nparallel, int epochsize, bool do_segmental_training, bool do_gradient_check, bool do_padding, bool b_use_additional_feature);


    void REINFORCEtrain(Model &model, Proc &am, Proc &am_agent_mirrow, Corpus &training, Corpus &devel, Trainer &sgd, string out_file, Dict & td, int max_epochs, int nparallel, cnn::real& largest_cost, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0);
    void REINFORCE_batch_train(Model &model, Proc &am, Proc &am_agent_mirrow,
        Corpus &training, Corpus &devel,
        Trainer &sgd, Dict& td, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
        bool sgd_update_epochs, bool do_gradient_check, bool b_inside_logic,
        cnn::real reward_baseline,
        cnn::real threshold_prob_for_sampling);
    void split_data_batch_reinforce_train(string train_filename, Model &model,
        Proc &hred, Proc & hred_agent_mirrow,
        Corpus &devel,
        Trainer &sgd, Dict& td,
        string out_file, 
        string model_file_name,
        int max_epochs, int nparallel, int epochsize,
        cnn::real & largest_cost, cnn::real reward_baseline, cnn::real threshold_prob,
        bool do_gradient_check);

    /** report perplexity

    @param words_s the word count in the source side
    @param words_t the word count in the target side

    @return entrpy loss
    */
    cnn::real testPPL(Model &model, Proc &am, Corpus &devel, NumTurn2DialogId& info, string out_file, bool segmental_training, cnn::real& words_s, cnn::real& words_t);
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo, bool segmental_training, const string& score_embedding_fn = "");
    void test_with_additional_feature(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test_segmental(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test(Model &model, Proc &am, TupleCorpus &devel, string out_file, Dict & sd, Dict & td);
    void testRanking(Model &, Proc &, Corpus &, Corpus &, string, Dict &, NumTurn2DialogId&, bool use_tfidf);

    void dialogue(Model &model, Proc &am, string out_file, Dict & td);

    void collect_sample_responses(Proc& am, Corpus &training);

    void nosegmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt,
        TrainingScores* scores, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr);
    void segmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt, TrainingScores *scores, bool resetmodel, bool doGradientCheck = false, Trainer* sgd = nullptr);
    pair<cnn::real, cnn::real> segmental_forward_backward_ranking(Model &model, Proc &am, PDialogue &v_v_dialogues, CandidateSentencesList &csls, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd);
    void segmental_forward_backward_with_additional_feature(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd);
    void REINFORCE_nosegmental_forward_backward(Model &model, Proc &am, Proc &am_mirrow, PDialogue &v_v_dialogues, int nutt,
        cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, Trainer* sgd, Dict& sd, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0,
        bool update_model = true);
    void REINFORCE_segmental_forward_backward(Proc &am, Proc &am_mirrow, PDialogue &v_v_dialogues, int nutt, Trainer* sgd, Dict& sd, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling, TrainingScores *scores, bool update_model);

public:
    /// for reranking
    bool MERT_tune(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    bool MERT_tune_edit_distance(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd, cnn::real weight_IDF=0.1);

public:
    /// for test ranking candidate
    /// @return a pair of numbers for top_1 and top_5 hits
    pair<unsigned, unsigned> segmental_forward_ranking(Model &model, Proc &am, PDialogue &v_v_dialogues, CandidateSentencesList &, int nutt, TrainingScores *scores, bool resetmodel, bool doGradientCheck = false, Trainer* sgd = nullptr);
    pair<unsigned, unsigned> segmental_forward_ranking_using_tfidf(Model &model, Proc &am, PDialogue &v_v_dialogues, CandidateSentencesList &, int nutt, TrainingScores *scores, bool resetmodel, bool doGradientCheck = false, Trainer* sgd = nullptr);

public:
    /// for LDA
    void lda_train(variables_map vm, const Corpus &training, const Corpus &test, Dict& sd);
    void lda_test(variables_map vm, const Corpus& test, Dict& sd);

public:
    /// for ngram
    void ngram_train(variables_map vm, const Corpus& test, Dict& sd);
    void ngram_clustering(variables_map vm, const Corpus& test, Dict& sd);
    void ngram_one_pass_clustering(variables_map vm, const Corpus& test, Dict& sd);
    void representative_presentation(
        vector<nGram> pnGram,
        const Sentences& responses,
        Dict& sd,
        vector<int>& i_data_to_cls,
        vector<string>& i_representative, cnn::real interpolation_wgt);
    void hierarchical_ngram_clustering(variables_map vm, const CorpusWithClassId& test, Dict& sd);
    int closest_class_id(vector<nGram>& pnGram, int this_cls, int nclsInEachCluster, const Sentence& obs, cnn::real& score, cnn::real interpolation_wgt);

public:
    /// compute tfidf weight for all words from training data
    /// dictionary or word list is given 
    /// TF : Term Frequency, which measures how frequently a term occurs in a document.Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones.Thus, the term frequency is often divided by the document length(aka.the total number of terms in the document) as a way of normalization :
    /// TF(t, d) = (Number of times term t appears in a document d) / (Total number of terms in the document).
    /// IDF : Inverse Document Frequency, which measures how important a term is.While computing TF, all terms are considered equally important.However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance.Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following :
    /// IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

    /// compute idf from training corpus, 
    /// exact tfidf score of a term needs to be computed given a sentence
    void get_idf(variables_map vm, const Corpus &training, Dict& sd);
protected:
    mutable vector<cnn::real> mv_idf; /// the dictionary for saving tfidf
    /// the index in this vector corresponds to index in the dictionary sd

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
    //test(model, am, devel, out_file + "bleu", sd);

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

    /// if report score in embedding space
    if (score_embedding_fn.size() > 0)
    {
        EvaluateProcess<AM_t> * ptr_evaluate = new EvaluateProcess<AM_t>();
        ptr_evaluate->readEmbedding(score_embedding_fn, td);

        cnn::real emb_loss = 0;
        cnn::real emb_chars_s = 0;
        cnn::real emb_chars_t = 0;
        cnn::real turns = 0;
        for (auto & diag : devel)
        {
            turns += ptr_evaluate->scoreInEmbeddingSpace(am, diag, td, emb_loss, emb_chars_s, emb_chars_t);
        }
        cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] word embedding loss = " << (emb_loss / turns) << " ppl=" << exp(emb_loss / turns) << ' ';
        of << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] word embedding loss = " << (emb_loss / turns) << " ppl=" << exp(emb_loss / turns) << endl;

        delete ptr_evaluate;
    }

    of.close();
}

/**
Test recall value 
*/
template <class AM_t>
void TrainProcess<AM_t>::testRanking(Model &model, AM_t &am, Corpus &devel, Corpus &train_corpus, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo,
    bool use_tfidf)
{
    unsigned lines = 0;
    unsigned hits_top_1 = 0;
    unsigned hits_top_5 = 0;

    map<int, tuple<int, int, int>> acc_over_turn;

    ofstream of(out_file);

    Timer iteration("completed in");

    dev_set_scores->reset();

    /// get all responses from training set, these responses will be used as negative samples
    Sentences negative_responses = get_all_responses(train_corpus);

    vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
    size_t id_stt_diag_id = 0;
    PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
    vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
    size_t ndutt = id_sel_idx.size();

    lines += ndutt * vd_dialogues.size();

    long rand_pos = 0;
    CandidateSentencesList csls = get_candidate_responses(vd_dialogues, negative_responses, rand_pos);

    while (ndutt > 0)
    {
        pair<unsigned, unsigned> this_hit;
        if (use_tfidf)
            this_hit = segmental_forward_ranking_using_tfidf(model, am, vd_dialogues, csls, ndutt, dev_set_scores, false);
        else
            this_hit = segmental_forward_ranking(model, am, vd_dialogues, csls, ndutt, dev_set_scores, false);
        
        hits_top_1 += this_hit.first;
        hits_top_5 += this_hit.second;

        if (acc_over_turn.find(vd_dialogues.size()) == acc_over_turn.end())
        {
            acc_over_turn[vd_dialogues.size()] = make_tuple(0, 0, 0);
        }
        get<0>(acc_over_turn[vd_dialogues.size()]) += this_hit.first;
        get<1>(acc_over_turn[vd_dialogues.size()]) += this_hit.second;
        get<2>(acc_over_turn[vd_dialogues.size()]) += ndutt * vd_dialogues.size();
        

        id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
        ndutt = id_sel_idx.size();
        lines += ndutt * vd_dialogues.size();

        csls = get_candidate_responses(vd_dialogues, negative_responses, rand_pos);

        if (verbose)
        {
            cerr << "selected " << ndutt << " :  ";
            for (auto p : id_sel_idx)
                cerr << p << " ";
            cerr << endl;
        }
    }


    for (auto iter = acc_over_turn.begin(); iter != acc_over_turn.end(); iter++)
    {
        auto key = iter->first;
        auto t = iter->second;

        cerr << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100<< "%." << endl;
        of << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100<< "%." << endl;
    }
    cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';
    of << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';

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
?So the BLEU score is artificially high
?However, because the use input is conditioned on the past response. If using the true decoder response as the past context, the user input cannot be from the corpus.
?Therefore, it is reasonable to use the true past response as context when evaluating the model.
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    BleuMetric bleuScore;
    bleuScore.Initialize();

    /*cnn::real idf_weight = 0.1;
    cnn::real edist_weight = 0.1;*/
    IDFMetric idfScore(mv_idf);

    EditDistanceMetric editDistScoreHyp;
    EditDistanceMetric editDistScoreRef;

    ofstream of(out_file);

    Timer iteration("completed in");

    for (auto diag : devel){

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        vector<vector<int>> res_kbest;
        vector<string> prv_response;
        vector<string> prv_response_ref;
        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> beam_search_results;

            if (turn_id == 0)
            {
                if (beam_search_decode == -1)
                    res = am.decode(turn.first, cg, sd);
                else
                    res = am.beam_decode(turn.first, cg, beam_search_decode, sd);
            }
            else
            {
                if (beam_search_decode == -1)
                    res = am.decode(prv_turn.second, turn.first, cg, sd);
                else
                    res = am.beam_decode(prv_turn.second, turn.first, cg, beam_search_decode, sd);
            }

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

            if (rerankIDF > 0)
            {
                beam_search_results = am.get_beam_decode_complete_list();

                /// averaged_log_likelihood , idf_score, bleu_score
                /// the goal is to rerank using averaged_log_likelihood + weight * idf_score
                /// so that the top is with the highest bleu score
                vector<int> best_res; 
                cnn::real largest_score = -10000.0; 
                while (!beam_search_results.empty())
                {
                    vector<int> result = beam_search_results.top().target;
                    cnn::real lk = beam_search_results.top().cost;
                    cnn::real idf_score = idfScore.GetStats(turn.second, result).second;

                    srec.clear();
                    for (auto p : result){
                        srec.push_back(sd.Convert(p));
                    }

                    cnn::real edist_score = editDistScoreHyp.GetStats(prv_response, srec);

                    cnn::real rerank_score = (1 - weight_IDF) * lk + weight_IDF * idf_score;

                    rerank_score = (1 - weight_edist) * rerank_score + weight_edist * edist_score;

                    if (rerank_score > largest_score)
                    {
                        largest_score = rerank_score;
                        best_res = result;
                    }
                    beam_search_results.pop();
                }

                if (best_res.size() > 0)
                {
                    srec.clear();
                    cout << "res response: ";
                    for (auto p : best_res){
                        cout << sd.Convert(p) << " ";
                        srec.push_back(sd.Convert(p));
                    }
                    cout << endl;
                }

                idfScore.AccumulateScore(turn.second, best_res);
            }
            else
            {
                if (res.size() > 0)
                {
                    cout << "res response: ";
                    for (auto p : res){
                        cout << sd.Convert(p) << " ";
                        srec.push_back(sd.Convert(p));
                    }
                    cout << endl;
                }

                idfScore.AccumulateScore(turn.second, res);
            }
            
            bleuScore.AccumulateScore(sref, srec);                


            if (turn_id > 0){
                editDistScoreHyp.AccumulateScore(prv_response, srec);
                editDistScoreRef.AccumulateScore(prv_response_ref, sref);
            }

            turn_id++;
            prv_turn = turn;
            prv_response = srec;
            prv_response_ref = sref;
        }
    }

    string sBleuScore = bleuScore.GetScore();
    cout << "BLEU (4) score = " << sBleuScore << endl;
    of << "BLEU (4) score = " << sBleuScore << endl;

    pair<cnn::real, cnn::real> idf_score = idfScore.GetScore();
    cout << "reference IDF = " << idf_score.first << " ; hypothesis IDF = " << idf_score.second << endl;
    of << "reference IDF = " << idf_score.first << " ; hypothesis IDF = " << idf_score.second << endl;

    cnn::real edit_distance_score_ref = editDistScoreRef.GetScore();
    cnn::real edit_distance_score_hyp = editDistScoreHyp.GetScore();
    cout << "average edit distance between two responses : reference: " << edit_distance_score_ref << " hypothesis: " << edit_distance_score_hyp << endl;
    of << "average edit distance between two responses : reference: " << edit_distance_score_ref << " hypothesis: " << edit_distance_score_hyp << endl;

    of.close();
}

template <class AM_t>
void TrainProcess<AM_t>::test_with_additional_feature(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    BleuMetric bleuScore;
    bleuScore.Initialize();

    /*cnn::real idf_weight = 0.1;
    cnn::real edist_weight = 0.1;*/
    IDFMetric idfScore(mv_idf);

    EditDistanceMetric editDistScoreHyp;
    EditDistanceMetric editDistScoreRef;

    ofstream of(out_file);

    Timer iteration("completed in");

    for (auto diag : devel){

        SentencePair prv_turn;
        SentencePair prv_turn_tfidf;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        vector<vector<int>> res_kbest;
        vector<string> prv_response;
        vector<string> prv_response_ref;
        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> beam_search_results;

            /// assign context
            if (turn_id == 0)
                prv_turn_tfidf = turn;
            else{
                prv_turn_tfidf.first.insert(prv_turn_tfidf.first.end(), turn.first.begin(), turn.first.end());
            }

            vector<cnn::real> reftfidf = ptr_tfidfScore->GetStats(prv_turn_tfidf.first);
            
            if (turn_id == 0)
            {
                prv_turn_tfidf = turn;

                if (beam_search_decode == -1)
                    res = am.decode_with_additional_feature(turn.first, reftfidf,cg, sd);
                else
                    res = am.beam_decode_with_additional_feature(turn.first, reftfidf, cg, beam_search_decode, sd);
            }
            else
            {
                if (beam_search_decode == -1)
                    res = am.decode_with_additional_feature(prv_turn.second, turn.first, reftfidf, cg, sd);
                else
                    res = am.beam_decode_with_additional_feature(prv_turn.second, turn.first, reftfidf, cg, beam_search_decode, sd);
            }

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

            if (rerankIDF > 0)
            {
                beam_search_results = am.get_beam_decode_complete_list();

                /// averaged_log_likelihood , idf_score, bleu_score
                /// the goal is to rerank using averaged_log_likelihood + weight * idf_score
                /// so that the top is with the highest bleu score
                vector<int> best_res;
                cnn::real largest_score = -10000.0;
                while (!beam_search_results.empty())
                {
                    vector<int> result = beam_search_results.top().target;
                    cnn::real lk = beam_search_results.top().cost;
                    cnn::real idf_score = idfScore.GetStats(turn.second, result).second;

                    srec.clear();
                    for (auto p : result){
                        srec.push_back(sd.Convert(p));
                    }

                    cnn::real edist_score = editDistScoreHyp.GetStats(prv_response, srec);

                    cnn::real rerank_score = (1 - weight_IDF) * lk + weight_IDF * idf_score;

                    rerank_score = (1 - weight_edist) * rerank_score + weight_edist * edist_score;

                    if (rerank_score > largest_score)
                    {
                        largest_score = rerank_score;
                        best_res = result;
                    }
                    beam_search_results.pop();
                }

                if (best_res.size() > 0)
                {
                    srec.clear();
                    cout << "res response: ";
                    for (auto p : best_res){
                        cout << sd.Convert(p) << " ";
                        srec.push_back(sd.Convert(p));
                    }
                    cout << endl;
                }

                idfScore.AccumulateScore(turn.second, best_res);
            }
            else
            {
                if (res.size() > 0)
                {
                    cout << "res response: ";
                    for (auto p : res){
                        cout << sd.Convert(p) << " ";
                        srec.push_back(sd.Convert(p));
                    }
                    cout << endl;
                }

                idfScore.AccumulateScore(turn.second, res);
            }

            bleuScore.AccumulateScore(sref, srec);


            if (turn_id > 0){
                editDistScoreHyp.AccumulateScore(prv_response, srec);
                editDistScoreRef.AccumulateScore(prv_response_ref, sref);
            }

            turn_id++;
            prv_turn = turn;
            prv_turn_tfidf.first.insert(prv_turn_tfidf.first.end(), turn.second.begin(), turn.second.end());
            prv_response = srec;
            prv_response_ref = sref;
        }
    }

    string sBleuScore = bleuScore.GetScore();
    cout << "BLEU (4) score = " << sBleuScore << endl;
    of << "BLEU (4) score = " << sBleuScore << endl;

    pair<cnn::real, cnn::real> idf_score = idfScore.GetScore();
    cout << "reference IDF = " << idf_score.first << " ; hypothesis IDF = " << idf_score.second << endl;
    of << "reference IDF = " << idf_score.first << " ; hypothesis IDF = " << idf_score.second << endl;

    cnn::real edit_distance_score_ref = editDistScoreRef.GetScore();
    cnn::real edit_distance_score_hyp = editDistScoreHyp.GetScore();
    cout << "average edit distance between two responses : reference: " << edit_distance_score_ref << " hypothesis: " << edit_distance_score_hyp << endl;
    of << "average edit distance between two responses : reference: " << edit_distance_score_ref << " hypothesis: " << edit_distance_score_hyp << endl;

    of.close();
}

/**
using beam search, generated candidate lists
each list has a tuple of scores
averaged_log_likelihood , idf_score, bleu_score

the goal of tuning is to rerank using averaged_log_likelihood + weight * idf_score
so that the top is with the highest bleu score

after tuning, the weight is computed and returned

@return weights
*/
template <class AM_t>
bool TrainProcess<AM_t>::MERT_tune(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    BleuMetric bleuScore;
    bleuScore.Initialize();
    IDFMetric idfScore(mv_idf);
    EditDistanceMetric editDistScoreRef;

    if (beam_search_decode <= 0)
    {
        cerr << "need beam search decoding to generate candidate lists. please set beamsearchdecode" << endl;
        return false;
    }

    if (rerankIDF != 1)
    {
      cerr << "need to set rerankIDF to 1" << endl;
      return false;
    }

    ofstream of(out_file);

    Timer iteration("completed in");

    map<cnn::real, cnn::real> weight_to_bleu_pair; 

    vector<vector<tuple<cnn::real, cnn::real, cnn::real>>> dev_set_rerank_scores;
    int samples = 0; 
    cout << "started decoding " << endl;

    for (auto diag : devel){

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        vector<vector<int>> res_kbest;
        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> beam_search_results;

            if (turn_id == 0)
	      res = am.beam_decode(turn.first, cg, beam_search_decode, sd);
            else
	      res = am.beam_decode(prv_turn.second, turn.first, cg, beam_search_decode, sd);

            sref.clear();
            if (turn.second.size() > 0)
            {
                for (auto p : turn.second){
                    sref.push_back(sd.Convert(p));
                }
            }

            if (rerankIDF > 0)
            {
                beam_search_results = am.get_beam_decode_complete_list();
		if (beam_search_results.empty())
		  cerr << "beam search complete list is empty " << endl;

                /// averaged_log_likelihood , idf_score, bleu_score
                /// the goal is to rerank using averaged_log_likelihood + weight * idf_score
                /// so that the top is with the highest bleu score
                vector<tuple<cnn::real, cnn::real, cnn::real>> rerank_scores;
                while (!beam_search_results.empty())
                {
                    vector<int> result = beam_search_results.top().target;
                    cnn::real lk = beam_search_results.top().cost;
                    cnn::real idf_score = idfScore.GetStats(turn.second, result).second;

                    srec.clear();
                    for (auto p : result){
                        srec.push_back(sd.Convert(p));
                    }

                    cnn::real bleu_score = bleuScore.GetSentenceScore(sref, srec);
                    beam_search_results.pop();

                    rerank_scores.push_back(make_tuple(lk, idf_score, bleu_score));
                }

                dev_set_rerank_scores.push_back(rerank_scores);
            }


            turn_id++;
            prv_turn = turn;
        }
        samples++;
        cout << " " << samples; 
        if (samples % 100 == 0)
            cout << "finished " << samples / (devel.size() + 0.0) * 100 << "%" << endl;
    }
    cout << "completed decoding" << endl;

    /// learn a weight to IDF score
    vector<cnn::real> v_bleu_scores; 
    vector<cnn::real> v_wgts;
    cnn::real max_bleu_score = -10000.0;
    int idx_wgt = -1;
    for (cnn::real idf_wgt = 0.0; idf_wgt <= 1.0; idf_wgt += 0.05)
    {
        v_wgts.push_back(idf_wgt);

        cnn::real avg_bleu_score = 0;
        for (auto t : dev_set_rerank_scores)
        {
            cnn::real max_combine_score = -10000.0;
            int idx = -1;
            int k = 0;
            for (auto c : t)
            {
                cnn::real lk = std::get<0>(c); 
                cnn::real idfscore = std::get<1>(c);
                cnn::real this_score = (1.0 - idf_wgt) * lk + idf_wgt * idfscore;
                if (max_combine_score < this_score)
                {
                    max_combine_score = this_score;
                    idx = k;
                }
                k++;
            }

            if (idx >= 0)
	      avg_bleu_score += std::get<2>(t[idx]);
	    else
	      cerr << "warning no bleu scores " << endl;
        }
        v_bleu_scores.push_back(avg_bleu_score / dev_set_rerank_scores.size());

        if (max_bleu_score < v_bleu_scores.back())
        {
	  max_bleu_score = v_bleu_scores.back();
	  idx_wgt = v_bleu_scores.size() - 1;
        }

	cout << "w(" << idf_wgt << ") " << v_bleu_scores.back() << " "; 
    }
    cout << endl;

    cnn::real optimal_wgt = v_wgts[idx_wgt];

    of << "optimal weight to IDF score is " << optimal_wgt << endl;
    cout << "optimal weight to IDF score is " << optimal_wgt << endl;

    return true;
}

template <class AM_t>
bool TrainProcess<AM_t>::MERT_tune_edit_distance(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd, cnn::real weight_IDF)
{
    BleuMetric bleuScore;
    bleuScore.Initialize();
    IDFMetric idfScore(mv_idf);
    EditDistanceMetric editDistScoreRef;

    if (beam_search_decode <= 0)
    {
        cerr << "need beam search decoding to generate candidate lists. please set beamsearchdecode" << endl;
        return false;
    }

    ofstream of(out_file);

    Timer iteration("completed in");

    map<cnn::real, cnn::real> weight_to_bleu_pair;

    vector<vector<tuple<cnn::real, cnn::real, cnn::real, cnn::real>>> dev_set_rerank_scores;
    int samples = 0;
    cout << "started decoding " << endl;

    for (auto diag : devel){

        Timer beam_decode("beam decode completed in");

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        vector<int> res;
        vector<vector<int>> res_kbest;
        vector<string> prv_response;

        for (auto spair : diag)
        {
            ComputationGraph cg;

            SentencePair turn = spair;
            vector<string> sref, srec;

            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> beam_search_results;

            if (turn_id == 0)
            {
                if (beam_search_decode == -1)
                    res = am.decode(turn.first, cg, sd);
                else
                    res = am.beam_decode(turn.first, cg, beam_search_decode, sd);
            }
            else
            {
                if (beam_search_decode == -1)
                    res = am.decode(prv_turn.second, turn.first, cg, sd);
                else
                    res = am.beam_decode(prv_turn.second, turn.first, cg, beam_search_decode, sd);
            }

            sref.clear();
            if (turn.second.size() > 0)
            {
                for (auto p : turn.second){
                    sref.push_back(sd.Convert(p));
                }
            }

            if (rerankIDF > 0)
            {
                beam_search_results = am.get_beam_decode_complete_list();

                /// averaged_log_likelihood , idf_score, bleu_score
                /// the goal is to rerank using averaged_log_likelihood + weight * idf_score
                /// so that the top is with the highest bleu score
                vector<tuple<cnn::real, cnn::real, cnn::real, cnn::real>> rerank_scores;
                while (!beam_search_results.empty())
                {
                    vector<int> result = beam_search_results.top().target;
                    cnn::real lk = beam_search_results.top().cost;
                    cnn::real idf_score = idfScore.GetStats(turn.second, result).second;

                    srec.clear();
                    for (auto p : result){
                        srec.push_back(sd.Convert(p));
                    }

                    cnn::real bleu_score = bleuScore.GetSentenceScore(sref, srec);
                    cnn::real edit_distance_score = editDistScoreRef.GetSentenceScore(prv_response, srec);

                    beam_search_results.pop();

                    rerank_scores.push_back(make_tuple(lk, idf_score, bleu_score, edit_distance_score));
                }

                dev_set_rerank_scores.push_back(rerank_scores);
            }


            turn_id++;
            prv_turn = turn;
            prv_response = srec;
        }
        samples++;
        cout << " " << samples;
        if (samples % 100 == 0)
            cout << "finished " << samples / (devel.size() + 0.0) * 100 << endl;
    }
    cout << "completed decoding" << endl;

    /// learn a weight to IDF score
    vector<cnn::real> v_bleu_scores;
    vector<cnn::real> v_wgts;
    cnn::real idf_wgt = weight_IDF;

    Timer MERT_tune("MERT tune completed in");

    for (cnn::real edst_wgt = 0.0; edst_wgt <= 0.2; edst_wgt += 0.005)
    {
        v_wgts.push_back(idf_wgt);

        cnn::real avg_bleu_score = 0;
        for (auto t : dev_set_rerank_scores)
        {
            cnn::real max_combine_score = -10000.0;
            int idx = -1;
            int k = 0;
            for (auto c : t)
            {
                cnn::real lk = std::get<0>(c);
                cnn::real idfscore = std::get<1>(c);
                cnn::real this_score = (1.0 - idf_wgt) * lk + idf_wgt * idfscore;
                cnn::real edistscore = std::get<3>(c);
                this_score = (1 - edst_wgt) * this_score + edst_wgt * edistscore;

                if (max_combine_score < this_score)
                {
                    max_combine_score = this_score;
                    idx = k;
                }
                k++;
            }

            avg_bleu_score += std::get<2>(t[idx]);
        }
        v_bleu_scores.push_back(avg_bleu_score / dev_set_rerank_scores.size());
    }

    cnn::real max_bleu_score = -10000.0;
    int idx_wgt = -1;
    cout << "bleu : ";
    for (int k = 0; k < v_bleu_scores.size(); k++)
    {
        if (max_bleu_score < v_bleu_scores[k])
        {
            max_bleu_score = v_bleu_scores[k];
            idx_wgt = k;
        }
        cout << v_bleu_scores[k] << " ";
    }
    cout << endl;

    cout << "weights : ";
    for (auto w : v_wgts)
        cout << w << " ";
    cout << endl;
    cnn::real optimal_wgt = v_wgts[idx_wgt];

    of << "optimal weight to IDF score is " << optimal_wgt << endl;
    cout << "optimal weight to IDF score is " << optimal_wgt << endl;

    return true;
}

/** warning, the test function use the true past response as the context, when measure bleu score
?So the BLEU score is artificially high
?However, because the use input is conditioned on the past response. If using the true decoder response as the past context, the user input cannot be from the corpus.
?Therefore, it is reasonable to use the true past response as context when evaluating the model.
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
            {
                if (beam_search_decode == -1)
                    res = am.decode(turn.first, cg, sd);
                else
                    res = am.beam_decode(turn.first, cg, beam_search_decode, sd);
            }
            else
            {
                if (beam_search_decode == -1)
                    res = am.decode(prv_turn.second, turn.first, cg, sd);
                else
                    res = am.beam_decode(prv_turn.second, turn.first, cg, beam_search_decode, sd);
            }

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

    IDFMetric idfScore(mv_idf);
    EditDistanceMetric editDistScoreHyp;

    int d_idx = 0;
    while (1){
        cout << "please start dialogue with the agent. you can end this dialogue by typing exit " << endl;

        size_t t_idx = 0;
        vector<int> decode_output;
        vector<int> shuman_input;
        Sentence prv_response;
        vector<string> prv_response_str;
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
            shuman = "<s> " + shuman + " </s>";
            convertHumanQuery(shuman, shuman_input, td);

            vector<string> sref, srec;

            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> beam_search_results;

            if (t_idx == 0)
            {
                if (beam_search_decode == -1)
                    decode_output = am.decode(shuman_input, cg, td);
                else
                    decode_output = am.beam_decode(shuman_input, cg, beam_search_decode, td);
            }
            else
            {
                if (beam_search_decode == -1)
                    decode_output = am.decode(prv_response, shuman_input, cg, td);
                else
                    decode_output = am.beam_decode(prv_response, shuman_input, cg, beam_search_decode, td);
            }

            if (rerankIDF > 0)
            {
                beam_search_results = am.get_beam_decode_complete_list();

                cnn::real largest_score = -10000.0;
                while (!beam_search_results.empty())
                {
                    vector<int> result = beam_search_results.top().target;
                    cnn::real lk = beam_search_results.top().cost;
                    cnn::real idf_score = idfScore.GetStats(result, result).second;

                    srec.clear();
                    for (auto p : result){
                        srec.push_back(sd.Convert(p));
                    }

                    cnn::real edit_distance_score = 0;
                    if (t_idx > 0)
                        edit_distance_score = editDistScoreHyp.GetSentenceScore(prv_response_str, srec);

                    beam_search_results.pop();

                    cnn::real score_combine_idf_lk = weight_IDF * idf_score + (1 - weight_IDF) * lk;
                    cnn::real comb_score = (1 - weight_edist) * score_combine_idf_lk + weight_edist * edit_distance_score;

                    if (comb_score > largest_score)
                    {
                        largest_score = comb_score;
                        decode_output = result;
                    }
                }
            }

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
            prv_response_str.clear();
            for (auto pp : decode_output)
            {
                cout << td.Convert(pp) << " ";
                prv_response_str.push_back(td.Convert(pp));
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

    IDFMetric idfScore(mv_idf);

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
                if (reinforceIDF <= 0)
                {
                    vector<string> sref, srec;
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

                    cnn::real score;  
                    score = bleuScore.GetSentenceScore(sref, srec);
                    v_bleu_score.push_back(score);
                }
                else
                {
                    vector<int> sref, srec;
                    if (verbose) cout << "ref response: ";
                    for (auto p : turn[k].second){
                        if (verbose) cout << sd.Convert(p) << " ";
                        sref.push_back(p);
                    }
                    if (verbose) cout << endl;


                    srec.clear();
                    if (verbose) cout << "res response: ";
                    for (auto p : q){
                        if (verbose) cout << sd.Convert(p) << " ";
                        srec.push_back(p);
                    }
                    if (verbose) cout << endl;

                    cnn::real score;
                    score = idfScore.GetSentenceScore(sref, srec).second;
                    v_bleu_score.push_back(score);
                    
                }

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
void TrainProcess<AM_t>::REINFORCE_segmental_forward_backward(AM_t &am, AM_t &am_mirrow, PDialogue &v_v_dialogues, int nutt, Trainer* sgd, Dict& sd, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling, TrainingScores *scores, bool update_model)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    PTurn prv_turn, new_turn, new_prv_turn;

    BleuMetric bleuScore;
    bleuScore.Initialize();

    IDFMetric idfScore(mv_idf);

    bool do_sampling = false;
    cnn::real rng_value = rand() / (RAND_MAX + 0.0);
    if (rng_value >= threshold_prob_for_sampling)
    {
        do_sampling = true;
    }

    am.reset();
    am_mirrow.reset();

    /// train on two segments of a dialogue
    vector<cnn::real> v_bleu_score;

    for (auto &turn : v_v_dialogues)
    {
        if (do_sampling)
        {
            vector<Sentence> res;
            ComputationGraph cg_sampling;
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
                res = am_mirrow.batch_decode(v_input, cg_sampling, sd);
            }
            else
            {
                res = am_mirrow.batch_decode(v_prv_response, v_input, cg_sampling, sd);
            }

            size_t k = 0;
            for (auto &q : res)
            {
                if (reinforceIDF <= 0)
                {
                    vector<string> sref, srec;
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

                    cnn::real score;
                    score = bleuScore.GetSentenceScore(sref, srec);
                    v_bleu_score.push_back(score);
                }
                else
                {
                    vector<int> sref, srec;
                    if (verbose) cout << "ref response: ";
                    for (auto p : turn[k].second){
                        if (verbose) cout << sd.Convert(p) << " ";
                        sref.push_back(p);
                    }
                    if (verbose) cout << endl;


                    srec.clear();
                    if (verbose) cout << "res response: ";
                    for (auto p : q){
                        if (verbose) cout << sd.Convert(p) << " ";
                        srec.push_back(p);
                    }
                    if (verbose) cout << endl;

                    cnn::real score;
                    score = idfScore.GetSentenceScore(sref, srec).second;
                    v_bleu_score.push_back(score);
                }

                k++;
            }

            /// use the decoded results as training signals
            /// reward uses either BLEU or IDF scores. these scores are associated with the decoded results
            /// training will encourage high BLEU or IDF scores, given these decoded results
            /// notice that BLEU score is a measure against true reference. therefore, the higher the BLEU 
            /// score, the closer or the better the decoded sequence is aligned to the reference.
            /// however, for IDF score, the highest IDF scores may correspond to many rare words that 
            /// don't make sense. 

            /// therefore, for IDF reward, we should use true reference as training signal
            /// in this case, the system is trained with references that have larger IDF values
            new_turn = turn;
            if (reinforceIDF <= 0)
            {
                /// this corresponds to using BLEU score, so should use decoded signal to encourage 
                /// learn from decoded context in order to generate high BLEU score outputs
                for (size_t k = 0; k < nutt; k++)
                {
                    new_turn[k].second = res[k];
                }
            }
            else
            {
                /// need to keep using the reference signals, as 
                /// high IDF doesn't mean good outputs
            }
        }
 
        /// graph for learning
        ComputationGraph cg;
        vector<Expression> v_errs; /// the errors to be minimized
        vector<Expression> i_err;

        /// get errors from the decoded results
        if (do_sampling)
        {
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

        Expression i_total_err = sum(v_errs);

        Tensor tv = cg.get_value(i_total_err);
        if (sgd != nullptr && update_model)
        {
            cg.backward();
            sgd->update(am.twords);
        }

        prv_turn = turn;
        new_prv_turn = new_turn;
        turn_id++;
        i_turns++;

        TensorTools::PushElementsToMemory(scores->training_score_current_location,
            scores->training_score_buf_size,
            scores->training_scores,
            tv);

        scores->swords += am.swords;
        scores->twords += am.twords;
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

		Tensor tv = cg.get_value(am.s2txent.i);
		TensorTools::PushElementsToMemory(scores->training_score_current_location,
			scores->training_score_buf_size,
			scores->training_scores,
			tv);

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

        scores->swords += am.swords;
        scores->twords += am.twords;

        prv_turn = turn;
        turn_id++;
        i_turns++;
    }
}

template <class AM_t>
void TrainProcess<AM_t>::segmental_forward_backward_with_additional_feature(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    PTurn prv_turn;
    PTurn prv_turn_tfidf;

    if (verbose)
        cout << "start segmental_forward_backward" << endl;

    for (auto turn : v_v_dialogues)
    {
        ComputationGraph cg;
        if (resetmodel)
        {
            am.reset();
        }

        /// assign context
        if (prv_turn_tfidf.size() == 0)
            prv_turn_tfidf = turn;
        else{
            for (int u = 0; u < nutt; u++)
            {
                prv_turn_tfidf[u].first.insert(prv_turn_tfidf[u].first.end(), turn[u].first.begin(), turn[u].first.end());
            }
        }

        vector<vector<cnn::real>> reftfidf_context;
        for (int u = 0; u < nutt; u++)
        {
            vector<cnn::real> reftfidf = ptr_tfidfScore->GetStats(prv_turn_tfidf[u].first);
            reftfidf_context.push_back(reftfidf);
        }

        if (turn_id == 0)
        {
            am.build_graph(turn, reftfidf_context, cg);
        }
        else
        {
            am.build_graph(prv_turn, turn, reftfidf_context, cg);
        }

        if (verbose) cout << "after graph build" << endl;

        Tensor tv = cg.get_value(am.s2txent.i);
        TensorTools::PushElementsToMemory(scores->training_score_current_location,
            scores->training_score_buf_size,
            scores->training_scores,
            tv);

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

        scores->swords += am.swords;
        scores->twords += am.twords;

        /// append this turn to context
        for (int i = 0; i < nutt; i++)
        {
            prv_turn_tfidf[i].first.insert(prv_turn_tfidf[i].first.end(), turn[i].second.begin(), turn[i].second.end());
        }
        prv_turn = turn;
        turn_id++;
        i_turns++;
    }
}

/**
return hit at rank0 (top-1) and hit within rank4 (top-5)
*/
template <class AM_t>
pair<cnn::real, cnn::real> TrainProcess<AM_t>::segmental_forward_backward_ranking(Model &model, AM_t &am, PDialogue &v_v_dialogues, CandidateSentencesList &csls, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    unsigned hits_top_5 = 0, hits_top_1 = 0;
    size_t num_candidate = MAX_NUMBER_OF_CANDIDATES;

    IDFMetric idfScore(mv_idf);

    PTurn prv_turn;

    if (verbose)
        cout << "start segmental_forward_backward" << endl;

    /// the negative candidate number should match to that expected
    assert(MAX_NUMBER_OF_CANDIDATES == csls[0].size());

    vector<vector<cnn::real>> correct_response_state;
    vector<vector<cnn::real>> prv_turn_correct_response_state;

    size_t idx = 0;
    for (auto turn : v_v_dialogues)
    {
        auto turn_back = turn;
        vector<vector<cnn::real>> costs(nutt, vector<cnn::real>(0));
        vector<vector<cnn::real>> correct_response_costs(nutt, vector<cnn::real>(0));

        /// first compute likelihoods from the correct paths
        {
            vector<Expression> v_errs;
            ComputationGraph cg;
            if (resetmodel)
            {
                am.reset();
            }

            if (turn_id == 0)
            {
                v_errs = am.build_graph(turn, cg);
            }
            else
            {
                am.copy_external_memory_to_cxt(cg, nutt, prv_turn_correct_response_state);  /// reset state to that coresponding to the correct response history for negative responses
                /// because this turn is dependent on the previous turn that is with the correct response

                v_errs = am.build_graph(prv_turn, turn, cg);
            }

            for (size_t err_idx = 0; err_idx < v_errs.size(); err_idx++)
            {
                Tensor tv = cg.get_value(v_errs[err_idx]);
                cnn::real lc = TensorTools::AccessElement(tv, 0) / turn[err_idx].second.size();
                cnn::real score = lc;
                correct_response_costs[err_idx].push_back(score);
            }
        }

        /// compute positive and negative sample's likelihoods
        for (int i = 0; i < num_candidate + 1; i++)
        {
            vector<Expression> v_errs;
            if (i < num_candidate)
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = csls[idx][i];
            }
            else
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = turn_back[ii].second;
            }

            ComputationGraph cg;
            if (resetmodel)
            {
                am.reset();
            }

            if (turn_id == 0)
            {
                v_errs = am.build_graph(turn, cg);
            }
            else
            {
                am.copy_external_memory_to_cxt(cg, nutt, prv_turn_correct_response_state);  /// reset state to that coresponding to the correct response history for negative responses
                /// because this turn is dependent on the previous turn that is with the correct response

                v_errs = am.build_graph(prv_turn, turn, cg);
            }

            if (verbose) cout << "after graph build" << endl;
            for (size_t err_idx = 0; err_idx < v_errs.size(); err_idx++)
            {
                Tensor tv = cg.get_value(v_errs[err_idx]);
                cnn::real score = TensorTools::AccessElement(tv, 0) / turn[err_idx].second.size();
                costs[err_idx].push_back(score);
            }

            if (sgd != nullptr)
            {
                /// compute average cost differences
                cnn::real cost_penalty = 0;
				int ndif = 0;
                for (size_t kk = 0; kk < v_errs.size(); kk++)
                {
                    cnn::real dif = correct_response_costs[kk].back() - costs[kk].back();
					if (dif > 0)
					{
						ndif++;
						cost_penalty += dif;
					}
                }

				if (i < num_candidate && cost_penalty > 0 || i == num_candidate)
				{
					if (verbose)
						cout << " start backprop " << endl;
					cg.backward();
					if (verbose)
						cout << " done backprop " << endl;

					cnn::real reward = 0.0;
					if (cost_penalty > 0 && i != num_candidate)
					{
						reward = -cost_penalty / ndif;
					}
					else
					{
						assert(i == num_candidate);
						reward = 1.0; /// this is the case of positive sample
					}
					if (verbose) 
						cout << "update model using reward " << cost_penalty << endl;
					sgd->update(am.twords, reward);  /// reinforce learning

					if (verbose)
						cout << " done update" << endl;
				}
				else{
					if (verbose)
						cout << "no need to update models" << endl;
				}
            }

            if (i == num_candidate)
            {
                /// this is the context with the correct responses history
                am.serialise_cxt_to_external_memory(cg, correct_response_state);
            }
        }

        prv_turn_correct_response_state = correct_response_state;

        for (size_t i = 0; i < costs.size(); i++)
        {
            vector<size_t> sorted_idx = sort_indexes<cnn::real>(costs[i]);
            vector<size_t>::iterator iter = find(sorted_idx.begin(), sorted_idx.end(), num_candidate);
            if (distance(iter, sorted_idx.end()) == 1)
            {
                hits_top_1++;
            }
            if (distance(iter, sorted_idx.end()) <= 5)
            {
                hits_top_5++;
            }

        }

        prv_turn = turn;
        turn_id++;
        i_turns++;
        idx++;
    }

    return make_pair(hits_top_1, hits_top_5);
}

/**
return hit at rank0 (top-1) and hit within rank4 (top-5)
*/
template <class AM_t>
pair<unsigned, unsigned> TrainProcess<AM_t>::segmental_forward_ranking(Model &model, AM_t &am, PDialogue &v_v_dialogues, CandidateSentencesList &csls, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    unsigned hits_top_5 = 0, hits_top_1 = 0;
    size_t num_candidate = MAX_NUMBER_OF_CANDIDATES;
    vector<Expression> v_errs;

    IDFMetric idfScore(mv_idf);

    PTurn prv_turn;
    PTurn prv_turn_tfidf;

    if (verbose)
        cout << "start segmental_forward_backward" << endl;

    /// the negative candidate number should match to that expected
    assert(MAX_NUMBER_OF_CANDIDATES == csls[0].size());

    vector<vector<cnn::real>> correct_response_state;
    vector<vector<cnn::real>> prv_turn_correct_response_state;

    size_t idx = 0;
    for (auto turn : v_v_dialogues)
    {
        auto turn_back = turn;
        vector<vector<cnn::real>> costs(nutt, vector<cnn::real>(0));

        /// assign context
        if (prv_turn_tfidf.size() == 0)
            prv_turn_tfidf = turn;
        else{
            for (int u = 0; u < nutt; u++)
            {
                prv_turn_tfidf[u].first.insert(prv_turn_tfidf[u].first.end(), turn[u].first.begin(), turn[u].first.end());
            }
        }

        vector<vector<cnn::real>> reftfidf_context;
        for (int u = 0; u < nutt; u++)
        {
            vector<cnn::real> reftfidf = ptr_tfidfScore->GetStats(prv_turn_tfidf[u].first);
            reftfidf_context.push_back(reftfidf);
        }

        for (int i = 0; i < num_candidate + 1; i++)
        {
            if (i < num_candidate)
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = csls[idx][i];
            }
            else
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = turn_back[ii].second;
            }

            ComputationGraph cg;
            if (resetmodel)
            {
                am.reset();
            }

            if (turn_id == 0)
            {
                am.copy_external_memory_to_cxt(cg, nutt, prv_turn_correct_response_state);
                v_errs = am.build_graph(turn, cg);
            }
            else
            {
                am.copy_external_memory_to_cxt(cg, nutt, prv_turn_correct_response_state);  /// reset state to that coresponding to the correct response history for negative responses
                /// because this turn is dependent on the previous turn that is with the correct response

                v_errs = am.build_graph(prv_turn, turn, cg);
            }

            if (verbose) cout << "after graph build" << endl;
            for (size_t err_idx = 0; err_idx < v_errs.size(); err_idx++)
            {
                Tensor tv = cg.get_value(v_errs[err_idx]);
                cnn::real lc = TensorTools::AccessElement(tv, 0) / turn[err_idx].second.size();
                cnn::real score = lc;
#ifdef RANKING_COMBINE_TFIDF
                vector<cnn::real> hyptfidf = ptr_tfidfScore->GetStats(turn[err_idx].second);
                /// compute cosine similarity
                cnn::real sim = cnn::metric::cosine_similarity(reftfidf_context[err_idx], hyptfidf);
                score = (1 - weight_IDF) * lc - weight_IDF * sim;
#endif

#ifdef RANKING_COMBINE_IDF
                cnn::real idf_score = idfScore.GetStats(turn[err_idx].first, turn[err_idx].second).second / turn[err_idx].second.size();
                score = (1 - weight_IDF) * lc - weight_IDF * idf_score;
#endif
                costs[err_idx].push_back(score);
            }

            if (i == num_candidate)
            {
                /// this is the context with the correct responses history
                am.serialise_cxt_to_external_memory(cg, correct_response_state);
                for (size_t k = 0; k < correct_response_state.size(); k++)
                for (size_t i = 0; i < correct_response_state[k].size(); i++)
                    correct_response_state[k][i] = 0;
            }
        }

        prv_turn_correct_response_state = correct_response_state;

        for (size_t i = 0; i < costs.size(); i++)
        {
            vector<size_t> sorted_idx = sort_indexes<cnn::real>(costs[i]);
            vector<size_t>::iterator iter = find(sorted_idx.begin(), sorted_idx.end(), num_candidate);
            if (distance(iter, sorted_idx.end()) == 1)
            {
                hits_top_1++;
            }
            if (distance(iter, sorted_idx.end()) <= 5)
            {
                hits_top_5++;
            }

        }

        prv_turn = turn;
        turn_id++;
        i_turns++;
        idx++;
    }

    return make_pair(hits_top_1, hits_top_5);
}

/**
return hit at rank0 (top-1) and hit within rank4 (top-5)
using tf-idf 
*/
template <class AM_t>
pair<unsigned, unsigned> TrainProcess<AM_t>::segmental_forward_ranking_using_tfidf(Model &model, AM_t &am, PDialogue &v_v_dialogues, CandidateSentencesList &csls, int nutt, TrainingScores * scores, bool resetmodel, bool doGradientCheck, Trainer* sgd)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    unsigned hits_top_5 = 0, hits_top_1 = 0;
    size_t num_candidate = MAX_NUMBER_OF_CANDIDATES;

    TFIDFMetric tfidfScore(mv_idf, sd.size());

    PTurn prv_turn;

    if (verbose)
        cout << "start segmental_forward_backward" << endl;

    /// the negative candidate number should match to that expected
    assert(MAX_NUMBER_OF_CANDIDATES == csls[0].size());

    size_t idx = 0;
    for (auto turn : v_v_dialogues)
    {
        auto turn_back = turn;
        vector<vector<cnn::real>> costs(nutt, vector<cnn::real>(0));

        /// assign context
        if (prv_turn.size() == 0)
            prv_turn= turn;
        else{
            for (int u = 0; u < nutt; u++)
            {
                prv_turn[u].first.insert(prv_turn[u].first.end(), turn[u].first.begin(), turn[u].first.end());
            }
        }

        /// all candidates have the same context
        vector<vector<cnn::real>> reftfidf_context; 
        for (int u = 0; u < nutt; u++)
        {
            vector<cnn::real> reftfidf = ptr_tfidfScore->GetStats(prv_turn[u].first);
            reftfidf_context.push_back(reftfidf);
        }
        
        for (int i = 0; i < num_candidate + 1; i++)
        {
            if (i < num_candidate)
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = csls[idx][i];
            }
            else
            {
                for (size_t ii = 0; ii < nutt; ii++)
                    turn[ii].second = turn_back[ii].second;
            }

            for (int u = 0; u < nutt; u++)
            {
                vector<cnn::real> hyptfidf = ptr_tfidfScore->GetStats(turn[u].second);
                /// compute cosine similarity
                cnn::real sim = cnn::metric::cosine_similarity(reftfidf_context[u], hyptfidf);
                cnn::real score = -sim; /// negative of similarity is cost

                costs[u].push_back(score);
            }

        }

        for (size_t i = 0; i < costs.size(); i++)
        {
            vector<size_t> sorted_idx = sort_indexes<cnn::real>(costs[i]);
            vector<size_t>::iterator iter = find(sorted_idx.begin(), sorted_idx.end(), num_candidate);
            if (distance(iter, sorted_idx.end()) == 1)
            {
                hits_top_1++;
            }
            if (distance(iter, sorted_idx.end()) <= 5)
            {
                hits_top_5++;
            }

        }

        /// append this turn to context
        for (int i = 0; i < nutt; i++)
        {
            prv_turn[i].first.insert(prv_turn[i].first.end(), turn[i].second.begin(), turn[i].second.end());
        }
        turn_id++;
        i_turns++;
        idx++;
    }

    return make_pair(hits_top_1, hits_top_5);
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

    save_cnn_model(out_file, &model);

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

                save_cnn_model(out_file, &model);
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
*/
template <class AM_t>
void TrainProcess<AM_t>::REINFORCE_batch_train(Model &model, AM_t &am, AM_t &am_agent_mirrow, 
    Corpus &training, Corpus &devel,
    Trainer &sgd, Dict& td, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
    bool sgd_update_epochs, bool do_gradient_check, bool b_inside_logic, 
    cnn::real reward_baseline, 
    cnn::real threshold_prob_for_sampling
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

            if (verbose)
            {
                cerr << "selected " << nutt << " :  ";
                for (auto p : i_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }

            REINFORCE_segmental_forward_backward(am, am_agent_mirrow, v_dialogues, nutt, &sgd, td, reward_baseline, threshold_prob_for_sampling, training_set_scores, true);

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

                save_cnn_model(out_file, &model);

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
            save_cnn_model(out_file + "e" + boost::lexical_cast<string>(sgd.epoch), &model);
        }
    }
}

/* the following does mutiple sentences per minibatch
@ b_inside_logic : use logic inside of batch to do evaluation on the dev set. if it is false, do dev set evaluation only if sgd.epoch changes
*/
template <class AM_t>
void TrainProcess<AM_t>::batch_train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
    bool sgd_update_epochs, bool do_gradient_check, bool b_inside_logic,
    bool b_do_padding, int kEOS, /// for padding if so use kEOS as the padding symbol
    bool b_use_additional_feature
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
                PDialogue pd = padding_with_eos(v_dialogues, kEOS, { false, true });
                v_dialogues = pd;
            }

            if (verbose)
            {
                cerr << "selected " << nutt << " :  ";
                for (auto p : i_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }


            if (b_use_additional_feature)
            {
                segmental_forward_backward_with_additional_feature(model, am, v_dialogues, nutt, training_set_scores, false, do_gradient_check, &sgd);
            }
            else
            {
                if (segmental_training)
                    segmental_forward_backward(model, am, v_dialogues, nutt, training_set_scores, false, do_gradient_check, &sgd);
                else
                    nosegmental_forward_backward(model, am, v_dialogues, nutt, training_set_scores, true, 0, &sgd);
            }
 
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

                save_cnn_model(out_file, &model);

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
            save_cnn_model(out_file + "e" + boost::lexical_cast<string>(sgd.epoch), &model);
        }
    }
}

/**
train ranking models
*/
template <class AM_t>
void TrainProcess<AM_t>::batch_train_ranking(Model &model, AM_t &am, size_t max_epochs, Corpus &train_corpus, string model_out_fn, string out_file, Dict & td, NumTurn2DialogId& train_corpusinfo, Trainer *sgd, int nparallel)
{
	if (train_corpus.size() == 0)
	{
		cerr << "no data for training" << endl;
		return;
	}

    unsigned lines = 0;
    unsigned hits_top_1 = 0;
    unsigned hits_top_5 = 0;

    map<int, tuple<int, int, int>> acc_over_turn;

    ofstream of(out_file);
	int ilines_check_point = 0; 
    Timer iteration("completed in");

    dev_set_scores->reset();

    /// get all responses from training set, these responses will be used as negative samples
    Sentences negative_responses = get_all_responses(train_corpus);

    vector<bool> vd_selected(train_corpus.size(), false);  /// track if a dialgoue is used
    size_t id_stt_diag_id = 0;
    PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
    vector<int> id_sel_idx = get_same_length_dialogues(train_corpus, nparallel, id_stt_diag_id, vd_selected, vd_dialogues, train_corpusinfo);
    size_t ndutt = id_sel_idx.size();

    lines += ndutt * vd_dialogues.size();

    long rand_pos = 0;
    CandidateSentencesList csls = get_candidate_responses(vd_dialogues, negative_responses, rand_pos);

    int train_epoch = 0;
    while (train_epoch < max_epochs)
    {
		hits_top_1 = 0;
		hits_top_5 = 0;
		acc_over_turn.clear();

        while (ndutt > 0)
        {
            pair<unsigned, unsigned> this_hit;
            this_hit = segmental_forward_backward_ranking(model, am, vd_dialogues, csls, ndutt, dev_set_scores, false, false, sgd);

            hits_top_1 += this_hit.first;
            hits_top_5 += this_hit.second;

            if (acc_over_turn.find(vd_dialogues.size()) == acc_over_turn.end())
            {
                acc_over_turn[vd_dialogues.size()] = make_tuple(0, 0, 0);
            }
            get<0>(acc_over_turn[vd_dialogues.size()]) += this_hit.first;
            get<1>(acc_over_turn[vd_dialogues.size()]) += this_hit.second;
            get<2>(acc_over_turn[vd_dialogues.size()]) += ndutt * vd_dialogues.size();


            id_sel_idx = get_same_length_dialogues(train_corpus, nparallel, id_stt_diag_id, vd_selected, vd_dialogues, train_corpusinfo);
            ndutt = id_sel_idx.size();
            lines += ndutt * vd_dialogues.size();

            csls = get_candidate_responses(vd_dialogues, negative_responses, rand_pos);

            if (verbose)
            {
                cerr << "selected " << ndutt << " :  ";
                for (auto p : id_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }

			ilines_check_point = lines;
			if (ilines_check_point > 50000)
			{
				save_cnn_model(model_out_fn + ".e" + boost::lexical_cast<string>(train_epoch) + ".ln" + boost::lexical_cast<string>(lines), &model);
				ilines_check_point = 0; 

        for (auto iter = acc_over_turn.begin(); iter != acc_over_turn.end(); iter++)
        {
            auto key = iter->first;
            auto t = iter->second;

            cerr << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100 << "%." << endl;
            of << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100 << "%." << endl;
        }
        cerr << "epoch " << train_epoch << "\n***Test [lines =" << lines << " out of total " << train_corpus.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';
        of << "epoch " << train_epoch << "\n***Test [lines =" << lines << " out of total " << train_corpus.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';

			}

        }


        for (auto iter = acc_over_turn.begin(); iter != acc_over_turn.end(); iter++)
        {
            auto key = iter->first;
            auto t = iter->second;

            cerr << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100 << "%." << endl;
            of << "turn len :" << key << ", " << get<2>(t) << "lines, R@1 " << get<0>(t) / (get<2>(t) +0.0) * 100 << "%., R@5 " << get<1>(t) / (get<2>(t) +0.0) * 100 << "%." << endl;
        }
        cerr << "epoch " << train_epoch << "\n***Test [lines =" << lines << " out of total " << train_corpus.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';
        of << "epoch " << train_epoch << "\n***Test [lines =" << lines << " out of total " << train_corpus.size() << " lines ] 1 in" << (MAX_NUMBER_OF_CANDIDATES + 1) << " R@1 " << hits_top_1 / (lines + 0.0) *100.0 << "%." << " R@5 " << hits_top_5 / (lines + 0.0) *100.0 << "%." << ' ';

        sgd->update_epoch();

		save_cnn_model(model_out_fn, &model);

        cerr << "**SHUFFLE\n";
        shuffle(training_numturn2did.vNumTurns.begin(), training_numturn2did.vNumTurns.end(), *rndeng);

        id_stt_diag_id = 0;
        vd_selected = vector<bool>(train_corpus.size(), false);
        for (auto p : training_numturn2did.mapNumTurn2DialogId){
            /// shuffle dailogues with the same number of turns
            random_shuffle(p.second.begin(), p.second.end());
        }

        vd_selected.assign(train_corpus.size(), false);

		train_epoch++;
		lines = 0;
    }

    of.close();
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, bool bcharlevel, bool nosplitdialogue)
{
    cnn::real best = std::numeric_limits<cnn::real>::max();
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    boost::mt19937 rng;                 // produces randomness out of thin air

    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;

    save_cnn_model(out_file, &model);

    reset_smoothed_ppl(ppl_hist);

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        training_set_scores->reset();
        dev_set_scores->reset();
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
            }

            // build graph for this instance
            auto& spair = training[order[si % order.size()]];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] << endl;

            /// find portion to train
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();
            if (nosplitdialogue)
                i_turn_to_train = 99999;

            vector<SentencePair> prv_turn;
            size_t turn_id = 0;

            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            do{
                ComputationGraph cg;

                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);
                for (size_t t = i_init_turn; t <= std::min(i_init_turn + i_turn_to_train, spair.size() - 1); t++)
                {
                    SentencePair turn = spair[t];
                    vector<SentencePair> i_turn(1, turn);
                    if (turn_id == 0)
                    {
                        am.build_graph(i_turn, cg);
                    }
                    else
                    {
                        am.build_graph(prv_turn, i_turn, cg);
                    }

                    turn_id++;

                    if (verbose)
                    {
                        display_value(am.s2txent, cg);
                        cnn::real tcxtent = as_scalar(cg.get_value(am.s2txent));
                        cerr << "xent = " << tcxtent << " nobs = " << am.twords << " PPL = " << exp(tcxtent / am.twords) << endl;
                    }

                    prv_turn = i_turn;
                    if (t == i_init_turn + i_turn_to_train || (t == spair.size() - 1)){

                        dloss += as_scalar(cg.get_value(am.s2txent.i));

                        dchars_s += am.swords;
                        dchars_t += am.twords;

                        cg.backward();
                        sgd.update(am.twords);

                        am.serialise_cxt(cg);
                        i_init_turn = t + 1;
                        i_turn_to_train = spair.size() - i_init_turn;
                        break;
                    }
                }
            } while (i_init_turn < spair.size());

            if (iter == report_every_i - 1)
                am.respond(spair, sd, bcharlevel);

            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';


        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (cnn::real)training.size()) == 0.0) {
            dev_set_scores->reset();

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
                    nosegmental_forward_backward(model, am, vd_dialogues, ndutt, dev_set_scores, true);

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

            dev_set_scores->compute_score();
            cnn::real ddloss = smoothed_ppl(dev_set_scores->dloss, ppl_hist);
            if (ddloss < best) {
                best = ddloss;

                save_cnn_model(out_file, &model);

            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dev_set_scores->dloss / dev_set_scores->twords) << " ppl=" << exp(dev_set_scores->dloss / dev_set_scores->twords) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}

/**
Training process on tuple corpus
*/
template <class AM_t>
void TrainProcess<AM_t>::train(Model &model, AM_t &am, TupleCorpus &training, Trainer &sgd, string out_file, int max_epochs)
{
    cnn::real best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    boost::mt19937 rng;                 // produces randomness out of thin air

    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    save_cnn_model(out_file, &model);

    reset_smoothed_ppl(ppl_hist);

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
            }

            // build graph for this instance
            auto& spair = training[order[si % order.size()]];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] << endl;

            /// find portion to train
            bool b_trained = false;
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();

            vector<SentenceTuple> prv_turn;
            size_t turn_id = 0;

            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            ComputationGraph cg;
            size_t t = 0;
            do{
                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);

                SentenceTuple turn = spair[t];
                vector<SentenceTuple> i_turn(1, turn);
                if (turn_id == 0)
                {
                    am.build_graph(i_turn, cg);
                }
                else
                {
                    am.build_graph(prv_turn, i_turn, cg);
                }

                turn_id++;

                t++;
                prv_turn = i_turn;
            } while (t < spair.size());

            dloss += as_scalar(cg.get_value(am.s2txent.i));

            dchars_s += am.swords;
            dchars_t += am.twords;

            //            CheckGrad(model, cg);
            cg.backward();
            sgd.update(am.twords);

            if (verbose)
                cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';
            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        if (fmod(lines, (cnn::real)training.size()) == 0)
        {
            cnn::real i_ppl = smoothed_ppl(exp(dloss / dchars_t), ppl_hist);
            if (best > i_ppl)
            {
                best = i_ppl;

                save_cnn_model(out_file, &model);
            }
            else
            {
                sgd.eta0 *= 0.5;
                sgd.eta *= 0.5;
            }
        }
        prv_epoch = floor(sgd.epoch);
    }
}

/**
collect sample responses
*/
template <class AM_t>
void TrainProcess<AM_t>::collect_sample_responses(AM_t& am, Corpus &training)
{
    am.clear_candidates();
    for (auto & ds : training){
        vector<SentencePair> prv_turn;

        for (auto& spair : ds){
            SentencePair turn = spair;
            am.collect_candidates(spair.second);
        }
    }
}

/**
overly pre-train models on small subset of the data
*/
template <class AM_t>
void TrainProcess<AM_t>::supervised_pretrain(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, cnn::real target_ppl, int min_diag_id,
    bool bcharlevel = false, bool nosplitdialogue = false)
{
    cnn::real best = std::numeric_limits<cnn::real>::max();
    unsigned report_every_i = 50;
    unsigned si = training.size(); /// number of dialgoues in training
    boost::mt19937 rng;                 // produces randomness out of thin air

    reset_smoothed_ppl(ppl_hist);

    size_t sample_step = 100;
    size_t maxepoch = sample_step * 10; /// no point of using more than 100 epochs, which correspond to use full data with 10 epochs for pre-train
    vector<unsigned> order(training.size() / sample_step);
    size_t k = 0;
    for (unsigned i = 0; i < training.size(); i += sample_step)
    {
        if (k < order.size())
            order[k++] = i;
        else
            break;
    }

    bool first = true;
    unsigned lines = 0;

    save_cnn_model(out_file, &model);

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (best > target_ppl && sgd.epoch < maxepoch) {
        Timer iteration("completed in");
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(order.size(), false);
            }

            // build graph for this instance
            auto& spair = training[order[si % order.size()] + min_diag_id];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] + min_diag_id << endl;

            /// find portion to train
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();
            if (nosplitdialogue)
                i_turn_to_train = 99999;

            vector<SentencePair> prv_turn;
            size_t turn_id = 0;

            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            do{
                ComputationGraph cg;

                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);
                for (size_t t = i_init_turn; t <= std::min(i_init_turn + i_turn_to_train, spair.size() - 1); t++)
                {
                    SentencePair turn = spair[t];
                    vector<SentencePair> i_turn(1, turn);
                    if (turn_id == 0)
                    {
                        am.build_graph(i_turn, cg);
                    }
                    else
                    {
                        am.build_graph(prv_turn, i_turn, cg);
                    }

                    turn_id++;

                    if (verbose)
                    {
                        display_value(am.s2txent, cg);
                        cnn::real tcxtent = as_scalar(cg.get_value(am.s2txent));
                        cerr << "xent = " << tcxtent << " nobs = " << am.twords << " PPL = " << exp(tcxtent / am.twords) << endl;
                    }

                    prv_turn = i_turn;
                    if (t == i_init_turn + i_turn_to_train || (t == spair.size() - 1)){

                        dloss += as_scalar(cg.get_value(am.s2txent.i));

                        dchars_s += am.swords;
                        dchars_t += am.twords;

                        cg.backward();
                        sgd.update(am.twords);

                        am.serialise_cxt(cg);
                        i_init_turn = t + 1;
                        i_turn_to_train = spair.size() - i_init_turn;
                        break;
                    }
                }
            } while (i_init_turn < spair.size());

            if (iter == report_every_i - 1)
                am.respond(spair, sd, bcharlevel);

            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)order.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        prv_epoch = floor(sgd.epoch);

        cnn::real i_ppl = smoothed_ppl(exp(dloss / dchars_t), ppl_hist);
        if (best > i_ppl)
        {
            best = i_ppl;
        }
        else
        {
            sgd.eta0 *= 0.5;
            sgd.eta *= 0.5;
        }
        if (sgd.eta < 1e-10)
        {
            cerr << "SGD stepsize is too small to update models" << endl;
            break;
        }
    }

    save_cnn_model(out_file, &model);

    save_cnn_model(out_file + ".pretrained", &model);

}

/**
online adaptation of an existing model
using only one sentence pair usually

in contrast, offline adaptation needs a corpus
*/
template <class AM_t>
void TrainProcess<AM_t>::online_adaptation(Model &model, AM_t &am,
    const Dialogue & training, // user_input_target_response_pair,
    Trainer &sgd, const cnn::real& target_ppl,
    int maxepoch,
    const string & updated_model_fname)
{
    cnn::real best = 9e+99;
    PDialogue ptraining;
    for (auto p : training)
    {
        PTurn pt(1);
        pt[0] = p;

        ptraining.push_back(pt);
    }

    while (best > target_ppl && sgd.epoch < maxepoch) {
        Timer iteration("completed in");
        training_set_scores->reset();

        segmental_forward_backward(model, am, ptraining, 1, training_set_scores, false, false, &sgd);

        sgd.status();
        training_set_scores->compute_score();

        cnn::real i_ppl = exp(training_set_scores->dloss / training_set_scores->twords);
        cerr << "\n***Train epoch[" << sgd.epoch << "] E = " << (training_set_scores->dloss / training_set_scores->twords) << " ppl=" << i_ppl << ' ';

        if (best > i_ppl)
        {
            best = i_ppl;
        }
        else
        {
            sgd.eta0 *= 0.5;
            sgd.eta *= 0.5;
        }
        if (sgd.eta < 1e-10)
        {
            cerr << "SGD stepsize is too small to update models" << endl;
            break;
        }
        sgd.update_epoch();
    }

    if (updated_model_fname.size() > 0)
        save_cnn_model(updated_model_fname, &model);
}

/**
since the tool loads data into memory and that can cause memory exhaustion, this function do sampling of data for each epoch.
*/
template <class AM_t>
void TrainProcess<AM_t>::split_data_batch_train(string train_filename, Model &model, AM_t &am, Corpus &devel,
    Trainer &sgd, string out_file,
    int max_epochs, int nparallel, int epochsize, bool segmental_training,
    bool do_gradient_check, bool do_padding, bool b_use_additional_feature)
{
    cnn::real largest_cost = std::numeric_limits<cnn::real>::max();
    cnn::real largest_dev_cost = std::numeric_limits<cnn::real>::max();

    reset_smoothed_ppl(ppl_hist);

    DataReader dr(train_filename);
    int trial = 0;
    dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);

    Corpus training = dr.corpus();
    training_numturn2did = get_numturn2dialid(training);

    save_cnn_model(out_file, &model);

    while (sgd.epoch < max_epochs)
    {
        Timer this_epoch("this epoch completed in");

        batch_train(model, am, training, devel, sgd, out_file, 1, nparallel, largest_cost, segmental_training, false, do_gradient_check, false, do_padding, kSRC_EOS, b_use_additional_feature);

        dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);
        training = dr.corpus();
        training_numturn2did = get_numturn2dialid(training);

        if (training.size() == 0)
        {
            dr.restart();
            dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);
            training = dr.corpus();  /// copy the data from data thread to the data to be used in the main thread
            training_numturn2did = get_numturn2dialid(training);
            //#define DEBUG
#ifndef DEBUG
            save_cnn_model(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), &model);
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

                    save_cnn_model(out_file, &model);
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

/**
since the tool loads data into memory and that can cause memory exhaustion, this function do sampling of data for each epoch.
*/
template <class AM_t>
void TrainProcess<AM_t>::split_data_batch_reinforce_train(string train_filename, Model &model, 
    AM_t &hred , AM_t& hred_agent_mirrow, 
    Corpus &devel,
    Trainer &sgd, Dict& td, 
    string out_file, string model_file_name,
    int max_epochs, int nparallel, int epochsize,
    cnn::real & largest_cost, cnn::real reward_baseline, cnn::real threshold_prob,
    bool do_gradient_check)
{
    long total_diags = 0;
    cnn::real largest_dev_cost = std::numeric_limits<cnn::real>::max();

    reset_smoothed_ppl(ppl_hist);

    DataReader dr(train_filename);
    int trial = 0;
    dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);

    Corpus training = dr.corpus();
    training_numturn2did = get_numturn2dialid(training);

    if (training.size() == 0)
    {
        cerr << "no content for " << train_filename << endl;
        throw("no content for training file");
    }

    while (sgd.epoch < max_epochs)
    {
        Timer this_epoch("this epoch completed in");
        REINFORCE_batch_train(model, hred, hred_agent_mirrow, 
            training, devel, sgd, td, out_file, 1, nparallel, largest_cost, false, false, do_gradient_check, false, 
            reward_baseline, threshold_prob);
    	total_diags += training.size();

        dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);
        training = dr.corpus();
        training_numturn2did = get_numturn2dialid(training);

	    /// save models for every batch of data
        save_cnn_model(model_file_name + ".i" + boost::lexical_cast<string>(sgd.epoch) + ".d" + boost::lexical_cast<string>(total_diags), &model);

        if (training.size() == 0)
        {
            dr.restart();
            dr.read_corpus(sd, kSRC_SOS, kSRC_EOS, epochsize);
            training = dr.corpus();  /// copy the data from data thread to the data to be used in the main thread
            training_numturn2did = get_numturn2dialid(training);
            sgd.update_epoch();

#ifndef DEBUG
            if (devel.size() > 0)
            {
                cnn::real ddloss, ddchars_s, ddchars_t;
                ddloss = testPPL(model, hred, devel, devel_numturn2did, out_file + ".dev.log", false, ddchars_s, ddchars_t);

                ddloss = smoothed_ppl(ddloss, ppl_hist);
                if (ddloss < largest_dev_cost) {
                    /// save the model with the best performance on the dev set
                    largest_dev_cost = ddloss;

                    save_cnn_model(model_file_name, &model);
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

template <class AM_t>
void TrainProcess<AM_t>::get_idf(variables_map vm, const Corpus &training, Dict& sd)
{
    long l_total_terms = 0;
    long l_total_nb_documents = 0;
    tWordid2TfIdf idf;

    for (auto & d : training)
    {
        for (auto &sp : d)
        {
            Sentence user = sp.first;
            Sentence resp = sp.second;

            l_total_terms += user.size();
            l_total_terms += resp.size();

            l_total_nb_documents += 2;

            tWordid2TfIdf occurence;
            for (auto& u : user)
            {
                occurence[u] = 1;
            }
            for (auto& u : resp)
            {
                occurence[u] = 1;
            }

            tWordid2TfIdf::iterator iter;
            for (iter = occurence.begin(); iter != occurence.end(); iter++)
                idf[iter->first] += 1;
        }
    }

    mv_idf.resize(sd.size(), 0);
    tWordid2TfIdf::iterator it;
    for (it = idf.begin(); it != idf.end(); it++)
    {
        cnn::real idf_val = it->second;
        idf_val = log(l_total_nb_documents / idf_val);

        int id = it->first;
        cnn::real idfscore = idf_val;
        mv_idf[id] = idfscore;
    }

    ptr_tfidfScore = new TFIDFMetric(mv_idf, sd.size());
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::lda_train(variables_map vm, const Corpus &training, const Corpus& test, Dict& sd)
{
    ldaModel * pLda = new ldaModel(training, test);

    pLda->init(vm);

    pLda->read_data(training, sd, test);

    pLda->train();
    pLda->save_ldaModel_topWords(vm["lda-model"].as<string>() + ".topic.words", sd);

    pLda->load_ldaModel(-1);
    pLda->test(sd);

    delete[] pLda;
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::lda_test(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    ldaModel * pLda = new ldaModel(empty, test);

    pLda->init(vm);
    pLda->load_ldaModel(vm["lda-final-model"].as<string>());

    pLda->read_data(empty, sd, test);

    pLda->test(sd);

    delete[] pLda;
}

/**
train n-gram model
*/
template <class AM_t>
void TrainProcess<AM_t>::ngram_train(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    nGram pnGram = nGram();
    pnGram.Initialize(vm);

    for (auto & t : test)
    {
        for (auto & s : t)
        {
            pnGram.UpdateNgramCounts(s.second, 0, sd);
            pnGram.UpdateNgramCounts(s.second, 1, sd);
        }
    }

    pnGram.ComputeNgramModel();

    pnGram.SaveModel();

}

/**
cluster using n-gram model
random shuffle training data and use the first half to train ngram model
after several iterations, which have their log-likelihood reported, the ngram model assign a class id for each sentence
*/
template <class AM_t>
void TrainProcess<AM_t>::ngram_clustering(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    int ncls = vm["ngram-num-clusters"].as<int>();
    vector<nGram> pnGram(ncls);
    for (auto& p : pnGram)
        p.Initialize(vm);

    cnn::real interpolation_wgt = vm["interpolation_wgt"].as<cnn::real>();

    /// flatten corpus
    Sentences order_kept_responses, response;
    flatten_corpus(test, order_kept_responses, response);
    order_kept_responses = response; ///

    vector<long> ncnt(ncls, 0); /// every class must have at least one sample
    int icnt = 0;

    for (int iter = 0; iter < vm["epochs"].as<int>(); iter++)
    {
        if (iter == 0)
        {
            shuffle(response.begin(), response.end(), std::default_random_engine(iter));
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].LoadModel(".m" + boost::lexical_cast<string>(i));
            }

            for (int i = 0; i < response.size(); i++)
            {
                /// every class has at least one sample
                int cls;
                if (icnt < ncls)
                {
                    if (ncnt[icnt] == 0)
                        cls = icnt++;
                    else
                        cls = rand0n_uniform(ncls - 1);
                }
                else
                    cls = rand0n_uniform(ncls - 1);

                pnGram[cls].UpdateNgramCounts(response[i], 0, sd);
                ncnt[cls] ++;
            }
#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].ComputeNgramModel();
                pnGram[i].SaveModel(".m" + boost::lexical_cast<string>(i));
            }

            std::fill(ncnt.begin(), ncnt.end(), 0);
        }
        else
        {
            /// use half the data to update centroids

            /// reassign data to closest cluster
            vector<Sentences> current_assignment(ncls);
            double totallk = 0;
            for (int i = 0; i < order_kept_responses.size(); i++)
            {
                if (order_kept_responses[i].size() == 0)
                    continue;

                cnn::real largest;
                int iarg = closest_class_id(pnGram, 0, ncls, order_kept_responses[i], largest, interpolation_wgt);

                current_assignment[iarg].push_back(order_kept_responses[i]);
                totallk += largest / order_kept_responses[i].size();
                ncnt[iarg]++;
            }
            totallk /= order_kept_responses.size();

            cout << "loglikelihood at iteration " << iter << " is " << totallk << endl;

            /// check if all clusters have at least one sample
            {
                int icls = 0;
                for (auto &p : ncnt)
                {
                    if (p < MIN_OCC_COUNT)
                    {
                        /// randomly pick one sample for this class
                        current_assignment[icls].push_back(response[rand0n_uniform(order_kept_responses.size()) - 1]);
                    }
                    icls++;
                }
            }
            std::fill(ncnt.begin(), ncnt.end(), 0);

            ///update cluster
#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
                pnGram[i].Clear();

            for (int i = 0; i < current_assignment.size(); i++)
            {
                for (auto & p : current_assignment[i])
                {
                    pnGram[i].UpdateNgramCounts(p, 0, sd);
                    pnGram[i].UpdateNgramCounts(p, 1, sd);
                }
            }

#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].ComputeNgramModel();
                pnGram[i].SaveModel(".m" + boost::lexical_cast<string>(i));
            }
        }
    }

    vector<int> i_data_to_cls;
    vector<string> i_represenative;
    representative_presentation(pnGram, order_kept_responses, sd, i_data_to_cls, i_represenative, interpolation_wgt);

    /// do classification now
    ofstream ofs;
    if (vm.count("outputfile") > 0)
        ofs.open(vm["outputfile"].as<string>());
    long did = 0;
    long idx = 0;
    for (auto& t : test)
    {
        int tid = 0;
        for (auto& s : t)
        {
            long iarg = i_data_to_cls[idx++];

            string userstr;
            for (auto& p : s.first)
                userstr = userstr + " " + sd.Convert(p);
            string responsestr;
            for (auto& p : s.second)
                responsestr = responsestr + " " + sd.Convert(p);

            string ostr = boost::lexical_cast<string>(did)+" ||| " + boost::lexical_cast<string>(tid)+" ||| " + userstr + " ||| " + responsestr;
            ostr = ostr + " ||| " + boost::lexical_cast<string>(iarg)+" ||| " + i_represenative[iarg];
            if (ofs.is_open())
            {
                ofs << ostr << endl;
            }
            else
                cout << ostr << endl;

            tid++;
        }
        did++;
    }

    if (ofs.is_open())
        ofs.close();
}

/**
keep turn id
each turn has its own clusters
ngram counts are not reliable. so use edit distance
*/
template <class AM_t>
void TrainProcess<AM_t>::ngram_one_pass_clustering(variables_map vm, const Corpus& test, Dict& sd)
{
#define MAXTURNS 100
    Corpus empty;
    int ncls = vm["ngram-num-clusters"].as<int>();
    vector<int> data2cls;
    vector<cnn::real> cls2score; /// the highest score in this class
    vector<Sentence> cls2data;  /// class to its closest response

    vector<int> cls_cnt;
    vector<nGram> pnGram;

    cnn::real threshold = vm["llkthreshold"].as<cnn::real>();

    cnn::real interpolation_wgt = vm["interpolation_wgt"].as<cnn::real>();

    vector<long> ncnt(MAXTURNS, 0); /// every class must have at least one sample

    long sid = 0;
    for (auto& d : test)
    {
        for (auto &t : d)
        {
            Sentence rep = remove_first_and_last(t.second);

            cnn::real largest = LZERO;
            int iarg;

            if (pnGram.size() > 0)
                iarg = closest_class_id(pnGram, 0, pnGram.size(), rep, largest, interpolation_wgt);

            if (largest < threshold && pnGram.size() < ncls)
            {
                pnGram.push_back(nGram());
                pnGram.back().Initialize(vm);
                pnGram.back().UpdateNgramCounts(rep, 0, sd);
                pnGram.back().UpdateNgramCounts(rep, 1, sd);
                iarg = pnGram.size() - 1;
            }
            else{
                pnGram[iarg].UpdateNgramCounts(rep, 0, sd);
                pnGram[iarg].UpdateNgramCounts(rep, 1, sd);
            }

            /// update centroid periodically
            if ((pnGram.size() < ncls) || ((pnGram.size() > ncls - 1) && sid % 1000 == 0))
            {
                for (auto &p : pnGram){
                    p.ComputeNgramModel();
                }
            }

            sid++;
        }
    }

    /// do classification now
    vector<Sentence> typical_response(pnGram.size());
    vector<cnn::real> best_score(pnGram.size(), LZERO);

    ofstream ofs;
    if (vm.count("outputfile") > 0)
        ofs.open(vm["outputfile"].as<string>());
    long did = 0;
    long idx = 0;
    for (auto& t : test)
    {
        int tid = 0;
        for (auto& s : t)
        {
            cnn::real largest;
            Sentence rep = remove_first_and_last(s.second);
            long iarg = closest_class_id(pnGram, 0, pnGram.size(), rep, largest, interpolation_wgt);

            if (best_score[iarg] < largest)
            {
                best_score[iarg] = largest;
                typical_response[iarg] = s.second;
            }
            data2cls.push_back(iarg);
        }
    }

    vector<string> i_representative;
    for (auto& p : typical_response)
    {
        string sl = "";
        for (auto w : p)
            sl = sl + sd.Convert(w) + " ";
        i_representative.push_back(sl);
    }

    long dataid = 0;
    did = 0;
    for (auto& t : test)
    {
        int tid = 0;
        for (auto& s : t)
        {
            string userstr;
            for (auto& p : s.first)
                userstr = userstr + " " + sd.Convert(p);
            string responsestr;
            for (auto& p : s.second)
                responsestr = responsestr + " " + sd.Convert(p);

            int iarg = data2cls[dataid];
            string ostr = boost::lexical_cast<string>(did)+" ||| " + boost::lexical_cast<string>(tid)+" ||| " + userstr + " ||| " + responsestr;
            ostr = ostr + " ||| " + boost::lexical_cast<string>(iarg)+" ||| " + i_representative[iarg];
            if (ofs.is_open())
            {
                ofs << ostr << endl;
            }
            else
                cout << ostr << endl;

            tid++;
            dataid++;
        }
        did++;
    }

    if (ofs.is_open())
        ofs.close();
}

/**
obtain the closet class id. the class has been orgnaized linearly
for example, the following is a vector of two large cluster, and there are three subclasses within each cluster
[cls_01 cls_02 cls_03 cls_11 cls_12 cls_13]
return the index, base 0, of the position of the class that has the largest likelihood.
The index is an absolution position, with offset of the base class position.
*/
template<class AM_t>
int TrainProcess<AM_t>::closest_class_id(vector<nGram>& pnGram, int this_cls, int nclsInEachCluster, const Sentence& obs,
    cnn::real& score, cnn::real interpolation_wgt)
{
    vector<cnn::real> llk(nclsInEachCluster);
    for (int c = 0; c < nclsInEachCluster; c++)
        llk[c] = pnGram[c + this_cls * nclsInEachCluster].GetSentenceLL(obs, interpolation_wgt);

    cnn::real largest = llk[0];
    int iarg = 0;
    for (int c = 1; c < nclsInEachCluster; c++)
    {
        if (llk[c] > largest)
        {
            largest = llk[c];
            iarg = c;
        }
    }
    score = largest;
    return iarg + this_cls;
}
/**
Find the top representative in each class and assign a representative, together with its index, to the original input
*/
template <class AM_t>
void TrainProcess<AM_t>::representative_presentation(
    vector<nGram> pnGram,
    const Sentences& responses,
    Dict& sd,
    vector<int>& i_data_to_cls,
    vector<string>& i_representative,
    cnn::real interpolation_wgt)
{

    long did = 0;
    int ncls = pnGram.size();
    vector<cnn::real> i_so_far_largest_score(ncls, -10000.0);/// the vector saving the largest score of a cluster from any observations so far
    vector<int> i_the_closet_input(ncls, -1); /// the index to the input that has the closest distance to centroid of each class
    i_representative.resize(ncls, "");

    for (auto& t : responses)
    {
        cnn::real largest;
        int iarg = closest_class_id(pnGram, 0, ncls, t, largest, interpolation_wgt);
        i_data_to_cls.push_back(iarg);

        long icls = 0;
        for (auto& p : pnGram)
        {
            cnn::real lk = p.GetSentenceLL(t, interpolation_wgt);
            if (i_so_far_largest_score[icls] < lk)
            {
                i_so_far_largest_score[icls] = lk;
                i_the_closet_input[icls] = i_data_to_cls.size() - 1;
            }
            icls++;
        }
    }

    /// represent the cluster with closest observation
    int i_representations = 0;
    for (int i = 0; i < ncls; i++)
    {
        i_representative[i] = "";
        if (i_the_closet_input[i] >= 0)
        {
            for (auto& p : responses[i_the_closet_input[i]])
                i_representative[i] = i_representative[i] + " " + sd.Convert(p);
            i_representations++;
        }
        else{
            cout << "cluster" << i << " is empty" << endl;
            throw("cluster is empty");
        }
    }

    cout << "total " << i_representations << " representations " << endl;
}

/**
Given trained model, do hierarchical ngram clustering
*/
template <class AM_t>
void TrainProcess<AM_t>::hierarchical_ngram_clustering(variables_map vm, const CorpusWithClassId& test, Dict& sd)
{
    Corpus empty;
    int ncls = vm["ngram-num-clusters"].as<int>();
    int nclsInEachCluster = vm["ncls-in-each-cluster"].as<int>();
    vector<nGram> pnGram(ncls*nclsInEachCluster);
    long i = 0;
    for (auto& p : pnGram)
    {
        p.Initialize(vm);
        p.LoadModel(".m" + boost::lexical_cast<string>(i));
        i++;
    }

    cnn::real interpolation_wgt = vm["interpolation_wgt"].as<cnn::real>();

    /// flatten corpus
    Sentences user_inputs;
    vector<SentenceWithId> response, not_randomly_shuffled_response;
    flatten_corpus(test, user_inputs, response);
    not_randomly_shuffled_response = response; /// backup of response that is not randomly shuffled
    user_inputs.clear();

    /// do classification now
    ofstream ofs;
    if (vm.count("outputfile") > 0)
        ofs.open(vm["outputfile"].as<string>());
    long did = 0;
    vector<cnn::real> i_so_far_largest_score(ncls * nclsInEachCluster, -10000.0);/// the vector saving the largest score of a cluster from any observations so far
    vector<int> i_the_closet_input(ncls * nclsInEachCluster, -1); /// the index to the input that has the closest distance to centroid of each class
    vector<int> i_data_to_cls;
    for (auto& t : test)
    {
        for (auto& s : t)
        {
            int this_cls = s.second.second;
            int cls_offset = this_cls * nclsInEachCluster;
            vector<cnn::real> llk(nclsInEachCluster);
            for (int i = 0; i < nclsInEachCluster; i++)
                llk[i] = pnGram[i + cls_offset].GetSentenceLL(s.second.first, interpolation_wgt);

            cnn::real largest = llk[0];
            int iarg = 0;
            for (int i = 1; i < nclsInEachCluster; i++)
            {
                if (llk[i] > largest)
                {
                    largest = llk[i];
                    iarg = i;
                }
            }
            iarg += cls_offset;
            i_data_to_cls.push_back(iarg);

            /// update representation of this class
            if (i_so_far_largest_score[iarg] < largest)
            {
                i_so_far_largest_score[iarg] = largest;
                i_the_closet_input[iarg] = i_data_to_cls.size() - 1;
            }
        }
    }

    /// represent the cluster with closest observation
    vector<string> i_representative(ncls * nclsInEachCluster);
    for (int i = 0; i < ncls *nclsInEachCluster; i++)
    {
        i_representative[i] = "";
        if (i_the_closet_input[i] >= 0)
        for (auto& p : not_randomly_shuffled_response[i_the_closet_input[i]].first)
            i_representative[i] = i_representative[i] + " " + sd.Convert(p);
    }

    long idx = 0;
    for (auto& t : test)
    {
        int tid = 0;
        for (auto& s : t)
        {
            int iarg = i_data_to_cls[idx];

            string userstr;
            for (auto& p : s.first)
                userstr = userstr + " " + sd.Convert(p);
            string responsestr;
            for (auto& p : s.second.first)
                responsestr = responsestr + " " + sd.Convert(p);

            string ostr = boost::lexical_cast<string>(did)+" ||| " + boost::lexical_cast<string>(tid)+" ||| " + userstr + " ||| " + responsestr;
            ostr = ostr + " ||| " + boost::lexical_cast<string>(iarg)+" ||| " + i_representative[iarg];
            if (ofs.is_open())
            {
                ofs << ostr << endl;
            }
            else
                cout << ostr << endl;

            tid++;
            idx++;
        }
        did++;
    }

    if (ofs.is_open())
        ofs.close();
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

public:
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
        Corpus training = read_corpus(ifs, sd, kSRC_SOS, kSRC_EOS, epochsize, make_pair<int, int>(2, 4), make_pair<bool, bool>(true, false),
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
            save_cnn_model(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), &model);

            sgd.update_epoch();
        }

        batch_train(model, am, training, devel, sgd, out_file, 1, nparallel, largest_cost, segmental_training, false, do_gradient_check, false);

        if (fmod(trial, 50) == 0)
        {
            save_cnn_model(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), &model);
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
                save_cnn_model(out_file, &model);
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
            save_cnn_model(out_file + "e" + boost::lexical_cast<string>(sgd.epoch), &model);
        }
    }
}


#endif
