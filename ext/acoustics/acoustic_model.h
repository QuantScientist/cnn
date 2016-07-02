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

    // interface for constructing an acoustic model
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
        int decoder_use_additional_input;

        // state variables used in the above two methods
        vector<Expression> v_src;
        Expression src;
        Expression i_sm0;  // the first input to decoder, even before observed
        std::vector<unsigned> src_len;
        Expression src_fwd;
        unsigned slen;

        // for initial hidden state
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
            const map<string, vector<unsigned>>& additional_vec_arguments,
            int decoder_use_additional_input, 
            cnn::real iscale = 1.0) :
            layers(layers),
            decoder(layers[DECODER_LAYER], vector<unsigned>{hidden_dims[DECODER_LAYER] + decoder_use_additional_input * hidden_dims[ENCODER_LAYER], hidden_dims[DECODER_LAYER], hidden_dims[DECODER_LAYER] }, &model, iscale),
            encoder_fwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            encoder_bwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            decoder_use_additional_input(decoder_use_additional_input),
            vocab_size_tgt(vocab_size_tgt)
        {
            hidden_dim = hidden_dims;

            p_cs = model.add_lookup_parameters(vocab_size_tgt, { hidden_dim[EMBEDDING_LAYER] }, iscale);
            p_R = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
            p_bias = model.add_parameters({ vocab_size_tgt }, iscale);

            p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], 2 * hidden_dim[ENCODER_LAYER] }, iscale);

            p_trns_src2hidden = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] }, iscale, "trans for obs to encoder");

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

    // interface for constructing an acoustic model
    template<class Builder, class Decoder>
    class CNNAcousticModel{
    protected:
        LookupParameters* p_cs;
        Parameters* p_bias;
        Parameters* p_R;  // for affine transformation after decoder
        Expression i_bias, i_R, i_bias_mb;

        vector<unsigned int> layers;
        vector<unsigned int> field_map_size; //number of filters at each layer
        vector<unsigned int> conv_block_x, conv_block_y; // receptive field dimension in x and y
        vector<unsigned int> stride_x, stride_y; // cnn' stride

        /// for alignment to source
        Parameters* p_U;
        Expression i_U;

        Model model;

        unsigned vocab_size_tgt;
        vector<unsigned int> hidden_dim;

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
    
        vector<Parameters*> p_filter; /// convolution filter
        vector<Expression> i_filter; 
        unsigned n_kernels;

        vector<Parameters*> p_conv_bias; /// convolution filter bias
        vector<Expression> i_conv_bias;

        vector<unsigned> pooling_kernel_x;
        vector<unsigned> pooling_kernel_y;

        Parameters * p_summary;  /// convert pooling output to output layer
        Expression i_summary;

        Parameters * p_merge;
        Expression i_merge;
    public:
        /// for criterion
        vector<Expression> v_errs;
        size_t src_feature_number;
        size_t tgt_words;

        vector<Expression> v_obs;

    public:
        cnn::real pr_threshold; /// probability threshold, above which an output is considered occured

    public:
        unsigned fd; /// source-side dimension after convoluion and pooling

    public:
        map<string, vector<unsigned>> additional_vec_arguments;

    public:
        CNNAcousticModel() {};
        CNNAcousticModel(cnn::Model& model, unsigned int vocab_size_tgt,
            const vector<unsigned int>& layers,
            const vector<unsigned int>& hidden_dims,
            const map<string, cnn::real>& additional_params,
            const map<string, vector<unsigned>>& additional_varguments, 
            int decoder_use_additional_input,
            cnn::real iscale = 1.0) :
            layers(layers),
            vocab_size_tgt(vocab_size_tgt),
            additional_vec_arguments(additional_varguments)
        {
            hidden_dim = hidden_dims;

            p_cs = model.add_lookup_parameters(vocab_size_tgt, { hidden_dim[EMBEDDING_LAYER] }, iscale);
            p_R = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
            p_bias = model.add_parameters({ vocab_size_tgt }, iscale);

            p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], 2 * hidden_dim[ENCODER_LAYER] }, iscale);

            p_trns_src2hidden = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] }, iscale, "trans for obs to encoder");

            zero.resize(hidden_dim[EMBEDDING_LAYER], 0);  /// for the no obs observation

            conv_block_x = additional_vec_arguments["conv_block_x"];
            conv_block_y = additional_vec_arguments["conv_block_y"];
            pooling_kernel_x = additional_vec_arguments["pooling_kernel_x"];
            pooling_kernel_y = additional_vec_arguments["pooling_kernel_y"];
            stride_x = additional_vec_arguments["stride_x"];
            stride_y = additional_vec_arguments["stride_y"];
            field_map_size = additional_vec_arguments["field_map_size"];

            unsigned input_dim = hidden_dim[EMBEDDING_LAYER];
            unsigned output_nfilter = 0;
            for (size_t i = 0; i < layers[ENCODER_LAYER]; i++)
            {
                /// embedding to encoding
                fd = ceil((input_dim - conv_block_y[i] + 1) / (stride_y[i] + 0.0));

                p_filter.push_back(model.add_parameters({ field_map_size[i], conv_block_x[i], conv_block_y[i] }, iscale));
                p_conv_bias.push_back(model.add_parameters({ 1 }, 0.0));

                if (fd > 1)
                    fd /= pooling_kernel_y[i];
                else
                {
                    output_nfilter = field_map_size[i];
                    break;
                }
                input_dim = fd;
            }

            p_emb2dec = model.add_parameters({ vocab_size_tgt, output_nfilter }, iscale);
            p_emb2dec_b = model.add_parameters({ vocab_size_tgt }, 0.0);

            pr_threshold = 0.0;
        };

        ~CNNAcousticModel(){
        };

        /// for context
        void reset()
        {
            turnid = 0;

            v_errs.clear();
            src_feature_number = 0;
            tgt_words = 0;

            v_obs.clear();
            i_filter.clear();
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

            if (i_filter.size() == 0)
            {
                i_filter.clear();
                for (auto p : p_filter)
                {
                    i_filter.push_back(parameter(cg, p)); 
                }

                for (auto p : p_conv_bias)
                {
                    i_conv_bias.push_back(parameter(cg, p));
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

            /// the source sentence has to be approximately the same length
            src_len = each_sentence_length(obs, hidden_dim[EMBEDDING_LAYER]);
            for (auto& p : src_len)
                src_feature_number += p * hidden_dim[EMBEDDING_LAYER];

            int ik_len = 0;
            unsigned input_dim = hidden_dim[EMBEDDING_LAYER];
            unsigned lc , ifd, ifilternbr; 
            for (auto&p : obs)
            {
                unsigned input_length = src_len[ik_len];
                Expression i_obs = input(cg, { hidden_dim[EMBEDDING_LAYER], src_len[ik_len] }, p);  /// d X T
                for (int lyr = 0; lyr < i_filter.size(); lyr++)
                {
                    Expression i_conv = conv2d(i_obs, i_filter[lyr], i_conv_bias[lyr], stride_x[lyr], stride_y[lyr]);
                    lc = ceil((input_length - conv_block_x[lyr] + 1) / (stride_x[lyr] + 0.0));
                    ifd = ceil((input_dim - conv_block_y[lyr] + 1) / (stride_y[lyr] + 0.0));
                    ifilternbr = field_map_size[lyr];

                    /// pooling without overlap
                    if (ifd > 1)
                    {
                        Expression i_pooled = max_pooling(i_conv, pooling_kernel_x[lyr], pooling_kernel_y[lyr], pooling_kernel_x[lyr], pooling_kernel_y[lyr]);
                        lc /= pooling_kernel_x[lyr];
                        if (lc <= 0)
                            throw("convolution has reduced to zero legth");

                        ifd /= pooling_kernel_y[lyr];
                        if (ifd <= 0)
                            throw("convolution has reduced to zero dimension");
                        i_obs = i_pooled;
                    }
                    else
                    {
                        i_obs = i_conv;
                    }
                    input_dim = ifd;
                    input_length = lc;

                    if (input_dim == 1)
                        break;
                }

                if (input_dim > 1)
                    throw("need more layers in order to reduce feature to 1 x sub-sampled-size x nfitler");
                vector<Expression> v_combine;
                Expression i_res = reshape(i_obs, {lc, ifilternbr});
                /// need something such as attention to pick the right input 
                /// at this moment, just max-pooling
                Expression i_pooled = max_pooling(i_res, lc, 1, lc-1, 1); 
                v_obs.push_back(reshape(i_pooled, { ifilternbr, 1 }));
                ik_len++;
            }
        };

        Expression decoder_step(ComputationGraph& cg)
        {
            Expression i_output = i_emb2dec * concatenate_cols(v_obs) + i_emb2dec_b;
            return i_output;
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

            Expression i_y_t = decoder_step(cg);
            Expression i_ydist = tanh(i_y_t);  /// values between -1 and 1
            Expression r_r_t = reshape(i_ydist, { vocab_size_tgt * nutt });

            for (size_t i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size_tgt;

                vector<cnn::real> v_tgt(vocab_size_tgt, -1.0); /// default is -1.0
                for (auto& lbl : lbls[i])
                {
                    v_tgt[lbl] = 1.0; /// target
                    tgt_words++;
                }

                Expression i_target = input(cg, { vocab_size_tgt },  v_tgt);
                Expression i_result = pickrange(r_r_t, offset, offset + vocab_size_tgt);
                this_errs[i].push_back(squared_distance(i_target, i_result)); 
            }

            for (auto &p : this_errs)
                errs.push_back(sum(p));

            turnid++;
            return errs;
        };

        std::vector<int> decode(const std::vector<cnn::real> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            nutt = 1;

            start_new_instance(vector<vector<cnn::real>>(nutt, source), cg);

            Expression i_y_t = decoder_step(cg);
            Expression i_ydist = tanh(i_y_t);
            Expression r_r_t = reshape(i_ydist, { vocab_size_tgt * nutt });

            auto dist = get_value(r_r_t, cg);
            vector<vector<int>> results(nutt);
            for (int u = 0; u < nutt; u++)
            {
                int i_shift = u * vocab_size_tgt;
                unsigned w = 0;
                for (unsigned x = 0; x < vocab_size_tgt; ++x) {
                    if (dist[x + i_shift] > pr_threshold) {
                        results[u].push_back(x);
                    }
                }
            }
            return results[0];
        }

    };

    // interface for constructing an acoustic model
    template<class Builder, class Decoder>
    class RNNAcousticModel{
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
        int decoder_use_additional_input;

        // state variables used in the above two methods
        vector<Expression> v_src;
        Expression src;
        Expression i_sm0;  // the first input to decoder, even before observed
        std::vector<unsigned> src_len;
        Expression src_fwd;
        unsigned slen;

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

        Parameters* p_filter; /// convolution filter
        Expression i_filter;

        Parameters * p_summary;  /// convert pooling output to output layer
        Expression i_summary;

        bool bidirectional;

    public:
        /// for criterion
        vector<Expression> v_errs;
        size_t src_feature_number;
        size_t tgt_words;

        vector<Expression> v_obs;

    public:
        cnn::real pr_threshold; /// probability threshold, above which an output is considered occured

    public:
        RNNAcousticModel() {};
        RNNAcousticModel(cnn::Model& model, unsigned int vocab_size_tgt,
            const vector<unsigned int>& layers,
            const vector<unsigned int>& hidden_dims,
            const map<string, cnn::real>& additional_params,
            const map<string, vector<unsigned>>& additional_vec_arguments,
            int decoder_use_additional_input,
            cnn::real iscale = 1.0) :
            layers(layers),
            decoder(layers[DECODER_LAYER], vector<unsigned>{hidden_dims[DECODER_LAYER] + decoder_use_additional_input * hidden_dims[ENCODER_LAYER], hidden_dims[DECODER_LAYER], hidden_dims[DECODER_LAYER] }, &model, iscale),
            encoder_fwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            encoder_bwd(layers[ENCODER_LAYER], vector<unsigned>{hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER]}, &model, iscale),
            decoder_use_additional_input(decoder_use_additional_input),
            vocab_size_tgt(vocab_size_tgt)
        {
            bidirectional = true;

            hidden_dim = hidden_dims;

            p_cs = model.add_lookup_parameters(vocab_size_tgt, { hidden_dim[EMBEDDING_LAYER] }, iscale);
            p_R = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
            p_bias = model.add_parameters({ vocab_size_tgt }, iscale);

            p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], 2 * hidden_dim[ENCODER_LAYER] }, iscale);

            p_trns_src2hidden = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] }, iscale, "trans for obs to encoder");

            zero.resize(hidden_dim[EMBEDDING_LAYER], 0);  /// for the no obs observation

            /// embedding to encoding
            p_emb2dec = model.add_parameters({ vocab_size_tgt, hidden_dim[ENCODER_LAYER] * (bidirectional?2:1)});
            p_emb2dec_b = model.add_parameters({ hidden_dim[ENCODER_LAYER] * (bidirectional ? 2 : 1) });

            /// calculate size for the RNN filter
            /// for simple RNN
            unsigned sz_filter = 0;
            /// first layer for input to hidden
            sz_filter += hidden_dim[EMBEDDING_LAYER] ; /// first layer linear transformation from input to hidden
            sz_filter += 1; /// first layer bias  
            /// first layer for recurrent connection
            sz_filter += hidden_dim[ENCODER_LAYER];  /// inside RNN that does hidden to hidden transformation
            sz_filter += 1;
            sz_filter *= hidden_dim[ENCODER_LAYER];
            sz_filter *= (bidirectional ? 2 : 1);

            for (int k = 1; k < layers[ENCODER_LAYER]; k++)
            {
                if (bidirectional)
                {
                    /// project to hidden_dim[ENCODER_LAYER] dimensioin
                    sz_filter += 2 * hidden_dim[ENCODER_LAYER] * hidden_dim[ENCODER_LAYER];
                }
                sz_filter += (bidirectional ? 2 : 1) * hidden_dim[ENCODER_LAYER] * hidden_dim[ENCODER_LAYER];  /// input to hidden
                sz_filter += (bidirectional ? 2 : 1) * hidden_dim[ENCODER_LAYER];
                sz_filter += (bidirectional ? 2 : 1) * hidden_dim[ENCODER_LAYER] * hidden_dim[ENCODER_LAYER];  /// hidden to hidden
                sz_filter += (bidirectional ? 2 : 1) * hidden_dim[ENCODER_LAYER];
            }
            p_filter = model.add_parameters({ sz_filter }, iscale);

            p_summary = model.add_parameters({ hidden_dim[DECODER_LAYER] }, iscale);

            pr_threshold = 0.0;
        };

        ~RNNAcousticModel(){
        };

        /// for context
        void reset()
        {
            turnid = 0;

            v_errs.clear();
            src_feature_number = 0;
            tgt_words = 0;

            v_obs.clear();
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

            {
                i_bias = parameter(cg, p_bias);
                i_R = parameter(cg, p_R);
                i_bias_mb = concatenate_cols(vector<Expression>(nutt, i_bias));

                trns_src2hidden = parameter(cg, p_trns_src2hidden);
                i_zero_emb = input(cg, { (hidden_dim[EMBEDDING_LAYER]) }, &zero);

                i_emb2dec = parameter(cg, p_emb2dec);
                i_emb2dec_b = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_emb2dec_b)));

                i_filter = parameter(cg, p_filter);

                i_summary = parameter(cg, p_summary);
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

            int ik_len = 0;
            bool for_training = true;
            for (auto&p : obs)
            {
                Expression i_obs = input(cg, { hidden_dim[EMBEDDING_LAYER], src_len[ik_len] }, p);
                int seqLength = src_len[ik_len];

                unsigned sz_hidden = layers[ENCODER_LAYER] * hidden_dim[ENCODER_LAYER] * 1 * (bidirectional ? 2 : 1);
                vector<cnn::real> init_hx(sz_hidden, 0.0);
                Expression i_hx = input(cg, { sz_hidden }, init_hx);
                Expression i_cx = input(cg, { sz_hidden }, init_hx);

                Expression i_rnn = cudnn_rnn(i_obs, i_filter, i_hx, i_cx, for_training, seqLength, nutt, hidden_dim[EMBEDDING_LAYER], hidden_dim[ENCODER_LAYER], layers[ENCODER_LAYER], bidirectional);
                Expression i_rnn_response = cudnn_seperate_outputs_from_hidden_states(i_rnn, hidden_dim[ENCODER_LAYER] * 1 * (bidirectional ? 2 : 1), seqLength, 1, layers[ENCODER_LAYER])[0];
                v_obs.push_back(i_rnn_response); /// a representation of this source
                ik_len++;
            }
        };

        Expression decoder_step(ComputationGraph& cg)
        {
            vector<Expression> v_x_t;
            /// get the last time outout
            int i = 0;
            for (auto& p : v_obs)
            {
                vector<cnn::real> mask(src_len[i], 0);
                mask[src_len[i]-1] = 1.0;
                Expression i_mask = input(cg, { src_len[i], 1 }, mask);
                Expression i_obs = p * i_mask; /// hidden_dim x 1
                v_x_t.push_back(i_obs);

                i++;
            }
            Expression i_output = i_emb2dec * (concatenate_cols(v_x_t) + i_emb2dec_b);
            return i_output;
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

            Expression i_y_t = decoder_step(cg);
            Expression i_ydist = tanh(i_y_t);  /// values between -1 and 1
            Expression r_r_t = reshape(i_ydist, { vocab_size_tgt * nutt });

            for (size_t i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size_tgt;

                vector<cnn::real> v_tgt(vocab_size_tgt, -1.0); /// default is -1.0
                for (auto& lbl : lbls[i])
                {
                    v_tgt[lbl] = 1.0; /// target
                    tgt_words++;
                }

                Expression i_target = input(cg, { vocab_size_tgt }, v_tgt);
                Expression i_result = pickrange(r_r_t, offset, offset + vocab_size_tgt);
                this_errs[i].push_back(squared_distance(i_target, i_result));
            }

            for (auto &p : this_errs)
                errs.push_back(sum(p));

            turnid++;
            return errs;
        };

        std::vector<int> decode(const std::vector<cnn::real> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            nutt = 1;

            start_new_instance(vector<vector<cnn::real>>(nutt, source), cg);

            Expression i_y_t = decoder_step(cg);
            Expression i_ydist = tanh(i_y_t);
            Expression r_r_t = reshape(i_ydist, { vocab_size_tgt * nutt });

            auto dist = get_value(r_r_t, cg);
            vector<vector<int>> results(nutt);
            for (int u = 0; u < nutt; u++)
            {
                int i_shift = u * vocab_size_tgt;
                unsigned w = 0;
                for (unsigned x = 0; x < vocab_size_tgt; ++x) {
                    if (dist[x + i_shift] > pr_threshold) {
                        results[u].push_back(x);
                    }
                }
            }
            return results[0];
        }

    };

} // namespace cnn

