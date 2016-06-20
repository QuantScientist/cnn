#pragma once

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/gru.h"
#include <algorithm>
#include <stack>
#include "cnn/data-util.h"

using namespace cnn::expr;
using namespace std;

namespace cnn {

    class Model;

    // interface for constructing an a dialogue 
    template<class Builder, class Decoder>
    class AcousticModel{
    protected:
        LookupParameters* p_cs;
        Parameters* p_bias;
        Parameters* p_R;  // for affine transformation after decoder
        Expression i_bias, i_R, i_bias_mb;

        vector<unsigned int> layers;
        Decoder decoder;  // for decoder at each turn
        Builder encoder_fwd, encoder_bwd; /// for encoder at each turn

        /// for alignment to source
        Parameters* p_U;
        Expression i_U;

        Model model;

        unsigned vocab_size_tgt;
        vector<unsigned int> hidden_dim;
        int rep_hidden;
        int decoder_use_additional_input;

        // state variables used in the above two methods
        vector<Expression> v_src;
        Expression src;
        Expression i_sm0;  // the first input to decoder, even before observed
        std::vector<unsigned> src_len;
        Expression src_fwd;
        unsigned slen;

        // for initial hidden state
        vector<Parameters*> p_h0;
        vector<Expression> i_h0;

        Parameters* p_trns_src2hidden;
        Expression trns_src2hidden;

        Parameters * p_emb2dec; /// embedding to encoding
        Parameters * p_emb2dec_b; /// bias 
        Expression   i_emb2dec;
        Expression   i_emb2dec_b;           /// the bias that is to be applied to each 
        size_t turnid;

        unsigned int nutt; // for multiple training utterance per inibatch

        vector<cnn::real> zero;
        Expression i_zero_emb; /// Expresison for embedding of zeros in the embedding space

    public:
        /// for criterion
        vector<Expression> v_errs;
        size_t src_feature_number;
        size_t tgt_words;

    public:
        AcousticModel() {};
        AcousticModel(cnn::Model& model, unsigned int vocab_size_tgt,
            const vector<unsigned int>& layers,
            const vector<unsigned int>& hidden_dims,
            const map<string, cnn::real>& additional_params,
            int hidden_replicates, 
            int decoder_use_additional_input, 
            cnn::real iscale = 1.0) :
            layers(layers),
            decoder(layers[DECODER_LAYER], vector<unsigned>{hidden_dims[DECODER_LAYER] + decoder_use_additional_input * hidden_dims[ENCODER_LAYER], hidden_dims[DECODER_LAYER], hidden_dims[DECODER_LAYER] }, &model, iscale),
            encoder_fwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            encoder_bwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            decoder_use_additional_input(decoder_use_additional_input),
            vocab_size_tgt(vocab_size_tgt),
            rep_hidden(hidden_replicates)
        {
            hidden_dim = hidden_dims;

            p_cs = model.add_lookup_parameters(vocab_size_tgt, { hidden_dim[EMBEDDING_LAYER] }, iscale);
            p_R = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
            p_bias = model.add_parameters({ vocab_size_tgt }, iscale);

            p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], 2 * hidden_dim[ENCODER_LAYER] }, iscale);

            p_trns_src2hidden = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] }, iscale, "trans for obs to encoder");

            for (size_t i = 0; i < layers[ENCODER_LAYER] * rep_hidden; i++)
            {
                p_h0.push_back(model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale));
                p_h0.back()->reset_to_zero();
            }
            zero.resize(hidden_dim[EMBEDDING_LAYER], 0);  /// for the no obs observation

            /// embedding to encoding
            p_emb2dec = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_dim[EMBEDDING_LAYER] });
            p_emb2dec_b = model.add_parameters({ hidden_dim[DECODER_LAYER] });

            i_h0.clear();
        };

        ~AcousticModel(){
        };

        /// for context
        void reset()
        {
            turnid = 0;

            i_h0.clear();

            v_errs.clear();
            src_feature_number = 0;
            tgt_words = 0;
        }

        void init_word_embedding(const map<int, vector<cnn::real>> & vWordEmbedding)
        {
            p_cs->copy(vWordEmbedding);
        }

        Expression sentence_embedding(const Sentence& sent, ComputationGraph& cg)
        {
            vector<Expression> vm;
            int t = 0;
            while (t < sent.size())
            {
                Expression xij = lookup(cg, p_cs, sent[t]);
                vm.push_back(xij);
                t++;
            }
            Expression i_x_t = average(vm);
            return i_x_t;
        }

    public:
        /// run in batch with multiple sentences
        /// source [utt][data stream] is utterance first and then its content
        /// the context RNN uses the last state of the encoder RNN as its input
        virtual void start_new_instance(
            const std::vector<std::vector<cnn::real>> &obs,
            ComputationGraph &cg)
        {
            nutt = obs.size();

            if (i_h0.size() == 0)
            {
                i_h0.clear();
                for (auto p : p_h0)
                {
                    i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
                }

                i_bias = parameter(cg, p_bias);
                i_R = parameter(cg, p_R);
                i_bias_mb = concatenate_cols(vector<Expression>(nutt, i_bias));

                trns_src2hidden = parameter(cg, p_trns_src2hidden);
                i_zero_emb = input(cg, { (hidden_dim[EMBEDDING_LAYER]) }, &zero);

                i_emb2dec = parameter(cg, p_emb2dec);
                i_emb2dec_b = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_emb2dec_b)));
            }

            size_t n_turns = 0;
            std::vector<Expression> source_embeddings;

            encoder_fwd.new_graph(cg);
            encoder_fwd.set_data_in_parallel(nutt);
            encoder_fwd.start_new_sequence();
            encoder_bwd.new_graph(cg);
            encoder_bwd.set_data_in_parallel(nutt);
            encoder_bwd.start_new_sequence();
            decoder.new_graph(cg);
            decoder.set_data_in_parallel(nutt);
            decoder.start_new_sequence();

            /// the source sentence has to be approximately the same length
            src_len = each_sentence_length(obs, hidden_dim[EMBEDDING_LAYER]);
            for (auto& p : src_len)
                src_feature_number += p * hidden_dim[EMBEDDING_LAYER];

            backward_directional(slen, obs, cg, zero,
                encoder_bwd, hidden_dim[EMBEDDING_LAYER], hidden_dim[ENCODER_LAYER], trns_src2hidden);

            //            v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

            //            i_U = parameter(cg, p_U);
            //           src = i_U * concatenate_cols(v_src);  // precompute 

            decoder.start_new_sequence(encoder_bwd.final_s());
        };

        /// support beam search decoder to pass previous state info
        Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg, RNNPointer *prev_state)
        {
            Expression i_c_t;
            unsigned int nutt = trg_tok.size();
            Expression i_h_t;  /// the decoder output before attention
            Expression i_h_attention_t; /// the attention state
            vector<Expression> v_x_t;

            if (verbose)
                cout << "AttMultiSource_LinearEncoder::decoder_step" << endl;
            for (auto p : trg_tok)
            {
                Expression i_x_x;
                if (p >= 0)
                    i_x_x = lookup(cg, p_cs, p);
                else
                    i_x_x = i_zero_emb;
                if (verbose)
                    display_value(i_x_x, cg, "i_x_x");
                v_x_t.push_back(i_x_x);
            }

            Expression i_obs = i_emb2dec * concatenate_cols(v_x_t) + i_emb2dec_b;
            Expression i_input;

            if (prev_state == nullptr)
            {
                i_input = i_obs; 
/*                if (attention_output_for_this_turn.size() <= turnid)
                {
                    i_input = i_obs;  //concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
                }
                else
                {
                    i_input = i_obs; // concatenate({ i_obs, attention_output_for_this_turn.back() });
                }
  */
            }
            else{
                NOT_IMPLEMENTED;
            }

            if (prev_state)
                i_h_t = decoder.add_input(*prev_state, i_input);
            else
                i_h_t = decoder.add_input(i_input);

            /// return the source side representation at output position after attention
/*            vector<Expression> alpha;
            vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

            /// compute response
            Expression concatenated_src = i_emb2enc * concatenate_cols(v_context_to_source) + i_emb2enc_b;
            Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
            i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

            attention_output_for_this_turn.push_back(i_h_attention_t);
*/
            Expression i_output = i_R * i_h_t;
//            Expression i_comb_max_entropy = i_output + i_max_ent_obs;

            return i_output + i_bias_mb;
        }

        vector<Expression> build_graph(
            const std::vector<std::vector<cnn::real>> &obs,
            const std::vector<std::vector<int>>& lbls,
            ComputationGraph &cg)
        {
            unsigned int nutt = obs.size();
            start_new_instance(obs, cg);

            vector<vector<Expression>> this_errs(nutt);
            vector<Expression> errs;

            nutt = lbls.size();

            int oslen = 0;
            for (auto p : lbls)
                oslen = (oslen < p.size()) ? p.size() : oslen;

            for (int t = 0; t < oslen; ++t) {
                vector<int> vobs;
                for (auto p : lbls)
                {
                    if (t < p.size())
                    {
                        vobs.push_back(p[t]);
                        tgt_words++;
                    }
                    else
                        vobs.push_back(-1);
                }
                Expression i_y_t = decoder_step(vobs, cg, nullptr);

                Expression i_ydist = log_softmax(i_y_t);
                Expression r_r_t = reshape(i_ydist, { vocab_size_tgt * nutt });

                for (size_t i = 0; i < nutt; i++)
                {
                    int offset = i * vocab_size_tgt;
                    if (t < lbls[i].size() - 1)
                    {
                        /// only compute errors on with output labels
                        this_errs[i].push_back(-pick(r_r_t, offset + lbls[i][t + 1]));
                    }
                }
            }

            for (auto &p : this_errs)
                errs.push_back(sum(p));

            turnid++;
            return errs;
        };
    
        std::vector<int> decode(const std::vector<cnn::real> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            const int sos_sym = tdict.Convert("S");
            const int eos_sym = tdict.Convert("!S");

            nutt = 1;

            std::vector<int> target;
            target.push_back(sos_sym);
            int t = 0;
            Sentence prv_response;

            start_new_instance(vector<vector<cnn::real>>(nutt, source), cg);

            while (target.back() != eos_sym)
            {
                vector<int> vobs(1, target.back());

                Expression i_y_t = decoder_step(vobs, cg, nullptr);

                Expression ydist = softmax(i_y_t);

                // find the argmax next word (greedy)
                unsigned w = 0;
                auto dist = get_value(ydist, cg); 
                auto pr_w = dist[w];
                for (unsigned x = 1; x < dist.size(); ++x) {
                    if (dist[x] > pr_w) {
                        w = x;
                        pr_w = dist[x];
                    }
                }

                // break potential infinite loop
                if (t > 100) {
                    w = eos_sym;
                    pr_w = dist[w];
                }

                t += 1;
                target.push_back(w);
            }

            return target;
        }

    };

} // namespace cnn

