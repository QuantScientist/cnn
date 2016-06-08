#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/rnnem.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"
#include "cnn/data-util.h"
#include "ext/dialogue/dialogue.h"
#include "cnn/macros.h"
//#include "ext/ir/ir.h"
#include "cnn/metric-util.h"
#include "ext/dialogue/attention_with_intention.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

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

using namespace std;

namespace cnn {

    template <class DBuilder>
    class DialogueProcessInfo{
    public:
        DBuilder s2tmodel;  /// source to target 

        int swords;
        int twords;
        int nbr_turns;

        Expression s2txent;

        DialogueProcessInfo(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates, 
            unsigned additional_input,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : s2tmodel(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
        {
            swords = 0;
            twords = 0;
            nbr_turns = 0;
        }
    
        // return Expression of total loss
        // only has one pair of sentence so far
        virtual vector<Expression> build_graph(const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

        // return Expression of total loss
        // only has one pair of sentence so far
        virtual vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            vector<int> results = s2tmodel.decode(source, cg, tdict);
            s2tmodel.serialise_context(cg);
            return results;
        }

        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            vector<int> results = s2tmodel.decode(cur, cg, tdict);
            s2tmodel.serialise_context(cg);
            return results;
        }

        // parallel decoding
        virtual vector<Sentence>batch_decode(const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict & tdict)
        {
            s2tmodel.reset();  /// reset network
            vector<Sentence> prv_sentence;
            vector<Sentence> iret = s2tmodel.batch_decode(prv_sentence, cur_sentence, cg, tdict);
            return iret;
        }

        virtual vector<Sentence> batch_decode(const vector<Sentence>& prv_sentence, const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict& tdict)
        {
            s2tmodel.assign_cxt(cg, cur_sentence.size());
            vector<Sentence> iret = s2tmodel.batch_decode(prv_sentence, cur_sentence, cg, tdict);
            return iret;
        }

        void assign_cxt(ComputationGraph& cg, size_t nutt)
        {
            twords = 0;
            swords = 0;
            s2tmodel.assign_cxt(cg, nutt);
        }

        void copy_external_memory_to_cxt(ComputationGraph& cg, size_t nutt, const vector<vector<cnn::real>>& external_state)
        {
            twords = 0;
            swords = 0;
            s2tmodel.copy_external_memory_to_cxt(cg, nutt, external_state);
        }

        void serialise_cxt(ComputationGraph& cg)
        {
            s2tmodel.serialise_context(cg);
        }

        void serialise_cxt_to_external_memory(ComputationGraph& cg, vector<vector<cnn::real>>& ext_memory)
        {
            s2tmodel.serialise_cxt_to_external_memory(cg, ext_memory);
        }

        void reset()
        {
            twords = 0;
            swords = 0;
            prv_response.clear();
            s2tmodel.reset();
        }

        void init_word_embedding(const map<int, vector<cnn::real>> &vWordEmbedding)
        {
            s2tmodel.init_word_embedding(vWordEmbedding);
        }

        void dump_word_embedding(const map<int, vector<cnn::real>>& vWordEmbedding, Dict& td, string ofn)
        {
            s2tmodel.dump_word_embedding(vWordEmbedding, td, ofn);
        }

        vector<cnn::real> sentence_embedding(const Sentence& sentence)
        {
            ComputationGraph cg;
            Expression emb = s2tmodel.sentence_embedding(sentence, cg); 
            return get_value(emb, cg);
        }

        cnn::real sentence_distance(const Sentence& sentencea, const Sentence& sentenceb)
        {
            ComputationGraph cg;
            Expression emba = s2tmodel.sentence_embedding(sentencea, cg);
            Expression embb = s2tmodel.sentence_embedding(sentenceb, cg);
            Expression dist = squared_distance(emba, embb);
            return get_value(dist, cg)[0];
        }

        void load_cls_info_from_file(string word2clsfn, string clsszefn, Dict& sd, Model& model)
        {
            s2tmodel.load_cls_info_from_file(word2clsfn, clsszefn, sd, model);
        }

        void collect_candidates(const std::vector<int>& response)
        {
            
        }

        void clear_candidates()
        {

        }

    public:

        /**
        @bcharlevel : true if output in character level, so not insert blank symbol after each output. default false.
        */
#ifdef INPUT_UTF8
        wstring respond(Model &model, wstring strquery, Dict<std::wstring>& td, bool bcharlevel = false)
#else
        string respond(Model &model, string strquery, Dict & td, 
            vector<vector<cnn::real>>& last_cxt_s,
            vector<vector<cnn::real>>& last_decoder_s,
            bool bcharlevel = false)
#endif
        {
#ifdef INPUT_UTF8
            wstring shuman;
            wstring response;
#else
            string shuman;
            string response;
#endif
            unsigned lines = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            shuman = "<s> " + strquery + " </s>";

            convertHumanQuery(shuman, shuman_input, td);

            ComputationGraph cg;
            if (prv_response.size() == 0)
                decode_output = decode(shuman_input, cg, td);
            else
                decode_output = decode(prv_response, shuman_input, cg, td);

            if (verbose)
            {
#ifdef INPUT_UTF8
                wcout << L"Agent: ";
                response = L"";
#else
                cout << "Agent: ";
                response = "";
#endif
            }

            for (auto pp : decode_output)
            {
                if (verbose)
                {
#ifdef INPUT_UTF8
                    wcout << td.Convert(pp) << L" ";
#else
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else
                        cout << td.Convert(pp);
#endif
                }

                if (pp != kSRC_EOS && pp != kSRC_SOS)
                    response = response + td.Convert(pp) + " ";

                if (verbose)
                {
                    if (!bcharlevel)
#ifdef INPUT_UTF8
                        wcout << L" ";
#else
                        cout << " ";
#endif
                }
            }

            if (verbose)
                cout << endl;

            prv_response = decode_output;
            return response; 
        }

        /// return levenshtein between responses and reference
        int respond(vector<SentencePair> diag, Dict & td, bool bcharlevel = false)
        {
            string shuman;
            string response;

            int iDist = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            prv_response.clear();

            ComputationGraph cg;

            for (auto p : diag)
            {
                if (prv_response.size() == 0)
                    decode_output = decode(p.first, cg, td);
                else
                    decode_output = decode(prv_response, p.first, cg, td);
                cout << "user : ";
                for (auto pp : p.first)
                {
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else 
                        cout << td.Convert(pp);
                }
                cout << endl;

                cout << "Agent: ";
                for (auto pp : decode_output)
                {
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else
                        cout << td.Convert(pp);
                }
                cout << endl;

                prv_response = decode_output;

                /// compute distance
                vector<string> sref;
                for (auto pp : p.second)
                    sref.push_back(td.Convert(pp)); 
                vector<string>sres;
                for (auto pp : decode_output)
                    sres.push_back(td.Convert(pp));

                iDist += cnn::metric::levenshtein_distance(sref, sres);
            }
            return iDist;
        }

        /// return levenshtein between responses and reference
        int respond(vector<SentencePair> diag, vector<SentencePair>& results, Dict & td)
        {
            string shuman;
            string response;

            int iDist = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            prv_response.clear();

            results.clear();
            for (auto p : diag)
            {
                ComputationGraph cg;

                SentencePair input_response;

                if (prv_response.size() == 0)
                    decode_output = decode(p.first, cg, td);
                else
                    decode_output = decode(prv_response, p.first, cg, td);
                cout << "user : ";
                for (auto pp : p.first)
                {
                    cout << td.Convert(pp) << " ";
                }
                cout << endl;

                cout << "Agent: ";
                for (auto pp : decode_output)
                {
                    cout << td.Convert(pp) << " ";
                }
                cout << endl;

                prv_response = decode_output;

                /// compute distance
                vector<string> sref;
                for (auto pp : p.second)
                    sref.push_back(td.Convert(pp));
                vector<string>sres;
                for (auto pp : decode_output)
                    sres.push_back(td.Convert(pp));

                input_response = make_pair(p.first, decode_output);
                results.push_back(input_response);

                iDist += cnn::metric::levenshtein_distance(sref, sres);
            }
            return iDist;
        }
    };

    template <class DBuilder>
    class MultiSourceDialogue: public DialogueProcessInfo<DBuilder>{
		public:
		using DialogueProcessInfo<DBuilder>::swords;		
		using DialogueProcessInfo<DBuilder>::twords;		
		using DialogueProcessInfo<DBuilder>::nbr_turns;		
		using DialogueProcessInfo<DBuilder>::s2txent;		
		using DialogueProcessInfo<DBuilder>::s2tmodel;		
        using DialogueProcessInfo<DBuilder>::serialise_cxt;
        using DialogueProcessInfo<DBuilder>::assign_cxt;

    public:
        explicit MultiSourceDialogue(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
            if (verbose)
                cout << "start MultiSourceDialogue:build_graph" << endl;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const Dialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;

            if (verbose)
                cout << "start MultiSourceDialogue:build_graph(const Dialogue & cur_sentence, ComputationGraph& cg)" << endl;

            twords = 0;
            swords = 0;
            nbr_turns = 1;
            vector<Sentence> insent, osent, prv_response;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);
            if (verbose)
                display_value(object.back(), cg, "object");

            s2txent = sum(object);
            if (verbose)
                display_value(s2txent, cg, "s2txent");

            s2tmodel.serialise_context(cg);

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return object;
        }

        /// for all speakers with history
        /// for feedforward network
        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent, prv_response;
            nbr_turns++;
            twords = 0;
            swords = 0;

            if (verbose)
                cout << "start MultiSourceDialogue:build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg)" << endl;

            for (auto p : prv_sentence)
            {
                prv_response.push_back(p.second);
            }

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.assign_cxt(cg, insent.size());
            vector<Expression> s2terr = s2tmodel.build_graph(prv_response, insent, osent, cg);
            Expression i_err = sum(s2terr);
            s2txent = i_err;
            if (verbose)
                display_value(s2txent, cg, "s2txent");

            s2tmodel.serialise_context(cg);

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return s2terr;
        }

        vector<Expression> build_graph(const Dialogue& cur_sentence,
            const vector<vector<cnn::real>>& additional_feature,
            ComputationGraph& cg)
        {
            return vector<Expression>();
        }

        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence,
            const vector<vector<cnn::real>>& additional_feature,
            ComputationGraph& cg) 
        {
            return vector<Expression>(); 
        }

        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode(source, cg, tdict);
        }

        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            return s2tmodel.decode(source, cur, cg, tdict);
        }

        virtual std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.sample(vector<int>(), source, cg, tdict);
        }

        virtual std::vector<int> sample(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            return s2tmodel.sample(source, cur, cg, tdict);
        }

        virtual std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.beam_decode(source, cg, beam_search_decode, tdict);
        }

        virtual std::vector<int> beam_decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            return s2tmodel.beam_decode(source, cur, cg, beam_search_decode, tdict);
        }

        virtual std::vector<int> decode_with_additional_feature(const std::vector<int> &source, const std::vector<cnn::real>&, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            return std::vector<int>(); 
        }

        virtual std::vector<int> decode_with_additional_feature(const std::vector<int> &source, const std::vector<int>& cur, 
            const std::vector<cnn::real>&, 
            ComputationGraph& cg, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        virtual std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &source, const std::vector<cnn::real>&,
            ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        virtual std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &source, const std::vector<int>& cur, 
            const std::vector<cnn::real>&, 
            ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> get_beam_decode_complete_list()
        {
            return s2tmodel.get_beam_decode_complete_list();
        }
    };

    template <class DBuilder>
    class ClassificationBasedMultiSourceDialogue : public DialogueProcessInfo<DBuilder>{
		public:
		using DialogueProcessInfo<DBuilder>::swords;		
		using DialogueProcessInfo<DBuilder>::twords;		
		using DialogueProcessInfo<DBuilder>::nbr_turns;		
		using DialogueProcessInfo<DBuilder>::s2txent;		
		using DialogueProcessInfo<DBuilder>::s2tmodel;		
    public:
        explicit ClassificationBasedMultiSourceDialogue(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
                if (verbose)
                    cout << "start MultiSourceDialogue:build_graph" << endl;
            }

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const Dialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;

            if (verbose)
                cout << "start MultiSourceDialogue:build_graph(const Dialogue & cur_sentence, ComputationGraph& cg)" << endl;

            twords = 0;
            swords = 0;
            nbr_turns = 1;
            vector<Sentence> insent, osent, prv_response;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size();
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);
            if (verbose)
                display_value(object.back(), cg, "object");

            s2txent = sum(object);
            if (verbose)
                display_value(s2txent, cg, "s2txent");

            assert(twords == s2tmodel.tgt_words);

            return object;
        }

        /// for all speakers with history
        /// for feedforward network
        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent, prv_response;
            nbr_turns++;
            twords = 0;
            swords = 0;

            if (verbose)
                cout << "start MultiSourceDialogue:build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg)" << endl;

            for (auto p : prv_sentence)
            {
                prv_response.push_back(p.second);
            }

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size();
                swords += p.first.size() - 1;
            }

            s2tmodel.assign_cxt(cg, insent.size());
            vector<Expression> s2terr = s2tmodel.build_graph(prv_response, insent, osent, cg);
            Expression i_err = sum(s2terr);
            s2txent = i_err;
            if (verbose)
                display_value(s2txent, cg, "s2txent");

            assert(twords == s2tmodel.tgt_words);

            return s2terr;
        }

        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode(source, cg, tdict);
        }

        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, source.size());
            return s2tmodel.decode(source, cur, cg, tdict);
        }

        /// return levenshtein between responses and reference
        int respond(vector<SentencePair> diag, vector<SentencePair>& results, Dict & td, stId2String<string>& responses)
        {
            string shuman;
            string response;

            unsigned lines = 0;

            int iDist = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            prv_response.clear();
            results.clear();
            
            for (auto p : diag)
            {
                ComputationGraph cg;
    
                SentencePair input_response;

                if (prv_response.size() == 0)
                    decode_output = decode(p.first, cg, td);
                else
                    decode_output = decode(prv_response, p.first, cg, td);
                cout << "user : ";
                for (auto pp : p.first)
                {
                    cout << td.Convert(pp) << " ";
                }
                cout << endl;

                cout << "Agent: ";
                for (auto pp : decode_output)
                {
                    int pid = responses.phyIdOflogicId(pp);
                    cout << responses.Convert(pid) << " ";
                }
                cout << endl;

                prv_response = decode_output;

                /// compute distance
                vector<string> sref;
                for (auto pp : p.second)
                    sref.push_back(td.Convert(pp));
                vector<string>sres;
                for (auto pp : decode_output)
                    sres.push_back(responses.Convert(pp));

                input_response = make_pair(p.first, decode_output);
                results.push_back(input_response);
            }
            return 0;
        }
    };

    /**
    Neural conversation model using sequence to sequence method
    arxiv.org/pdf/1506.05869v2.pdf
    */
    template <class DBuilder>
    class DialogueSeq2SeqModel : public DialogueProcessInfo<DBuilder> {
    private:
        vector<Expression> i_errs;

    public:
        DialogueSeq2SeqModel(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
        }


        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const vector<SentencePair> & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;
            vector<Sentence> insent, osent;

            i_errs.clear();

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);

            s2txent = sum(object);

            s2tmodel.serialise_context(cg);

            i_errs.push_back(s2txent);
            return object;
        }

        /**
        concatenate the previous response and the current source as input, and predict the reponse of the current turn
        */
        vector<Expression> build_graph(const vector<SentencePair>& prv_sentence, const vector<SentencePair>& cur_sentence, ComputationGraph& cg) override
        {
            swords = twords = 0;
            vector<Sentence> insent, osent;

            for (auto p : prv_sentence)
            {
                /// remove sentence ending
                Sentence i_s;
                for (auto & w : p.second){
                    if (w != kSRC_EOS)
                        i_s.push_back(w);
                }
                insent.push_back(i_s);
            }

            size_t k = 0;
            for (auto p : cur_sentence)
            {
                /// remove sentence begining
                for (auto & w : p.first){
                    if (w != kSRC_SOS)
                        insent[k].push_back(w);
                }
                swords += insent[k].size() - 1;
                k++;
            }

            for (auto p : cur_sentence)
            {
                osent.push_back(p.second);

                twords += p.second.size() - 1;
            }

            int nutt = cur_sentence.size();

            s2tmodel.assign_cxt(cg, nutt);
            vector<Expression> object_cur_s2cur_t = s2tmodel.build_graph(insent, osent, cg);
            Expression i_err = sum(object_cur_s2cur_t);

            i_errs.push_back(i_err);

            s2txent = i_err;

            s2tmodel.serialise_context(cg);

            return object_cur_s2cur_t;
        }

        vector<Expression> build_graph(const Dialogue & cur_sentence,
            const vector<vector<cnn::real>>& additional_feature,
            ComputationGraph& cg)
        {
            vector<Expression> object;
            return object;
        }
        
        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence,
            const vector<vector<cnn::real>>& additional_feature,
            ComputationGraph& cg)
        {
            vector<Expression> object;
            NOT_IMPLEMENTED;
            return object;
        }
        
        std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict) override
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode(source, cg, tdict);
        }

        std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict) override
        {
            Sentence insent;

            /// remove sentence ending
            for (auto & w : source){
                if (w != kSRC_EOS)
                    insent.push_back(w);
            }

            /// remove sentence begining
            for (auto & w : cur){
                if (w != kSRC_SOS)
                    insent.push_back(w);
            }

            swords += insent.size() - 1;

            s2tmodel.assign_cxt(cg, 1);
            vector<int> results = s2tmodel.decode(insent, cg, tdict);
            s2tmodel.serialise_context(cg);

            return results;
        }

        // parallel decoding
        vector<Sentence>batch_decode(const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict & tdict)
        {
            s2tmodel.reset();  /// reset network
            vector<Sentence> prv_sentence;
            vector<Sentence> iret = s2tmodel.batch_decode(prv_sentence, cur_sentence, cg, tdict);
            return iret;
        }

        vector<Sentence> batch_decode(const vector<Sentence>& prv_sentence, const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict& tdict)
        {
            vector<Sentence> insent;

            for (auto p : prv_sentence)
            {
                /// remove sentence ending
                Sentence i_s;
                for (auto & w : p){
                    if (w != kSRC_EOS)
                        i_s.push_back(w);
                }
                insent.push_back(i_s);
            }

            size_t k = 0;
            for (auto p : cur_sentence)
            {
                /// remove sentence begining
                for (auto & w : p){
                    if (w != kSRC_SOS)
                        insent[k].push_back(w);
                }
                swords += insent[k].size() - 1;
                k++;
            }

            vector<Sentence> iret = s2tmodel.batch_decode(prv_sentence, insent, cg, tdict);
            return iret;
        }
    
        virtual std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.sample(vector<int>(), source, cg, tdict);
        }

        virtual std::vector<int> sample(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            return s2tmodel.sample(source, cur, cg, tdict);
        }

        std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.beam_decode(source, cg, beam_search_decode, tdict);
        }

        std::vector<int> beam_decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            s2tmodel.assign_cxt(cg, 1);
            return s2tmodel.beam_decode(source, cur, cg, beam_search_decode, tdict);
        }

        std::vector<int> decode_with_additional_feature(const std::vector<int> &source, const std::vector<cnn::real>&, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        std::vector<int> decode_with_additional_feature(const std::vector<int> &source, const std::vector<int>& cur,
            const std::vector<cnn::real>&,
            ComputationGraph& cg, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &source, const std::vector<cnn::real>&,
            ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &source, const std::vector<int>& cur,
            const std::vector<cnn::real>&,
            ComputationGraph& cg, int beam_search_width, cnn::Dict  &tdict)
        {
            return std::vector<int>();
        }

        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> get_beam_decode_complete_list()
        {
            return s2tmodel.get_beam_decode_complete_list();
        }
    };

}; // namespace cnn
