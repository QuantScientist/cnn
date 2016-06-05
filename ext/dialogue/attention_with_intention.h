#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
//#include "rnnem.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"
#include "cnn/data-util.h"
#include "cnn/dnn.h"
#include "cnn/math.h"
//#include "cnn/decode.h"
//#include "rl.h"
#include "ext/dialogue/dialogue.h"
#include "cnn/approximator.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

extern int verbose; 
extern int max_number_of_hypothesis;

namespace cnn {

#define MEM_SIZE 10
#define REASONING_STEPS 7

    struct Hypothesis {
        Hypothesis(RNNPointer state, int tgt, cnn::real cst, int _t)
        : builder_state(state), target({ tgt }), cost(cst), t(_t) {}
        Hypothesis(RNNPointer state, int tgt, cnn::real cst, Hypothesis &last)
            : builder_state(state), target(last.target), cost(cst), t(last.t + 1) {
            target.push_back(tgt);
        }
        RNNPointer builder_state;
        std::vector<int> target;
        cnn::real cost;
        int t;
    };

    struct CompareHypothesis
    {
        bool operator()(const Hypothesis& h1, const Hypothesis& h2)
        {
            if (h1.cost < h2.cost) return true;
            return false;
        }
    };
/**
use simple models 
encoder:
prev_response -> vector representation -> 
current_user  -> vector representation -> 
                 previous intention hidden -> combine at each layer linearly (z_k = c z_k-1 + W_p y_k-1 + W_u x_k)
decoder:
simple RNN, with initial state from the above combination at each layer. 
or can be simpler such as bilinear model
*/
template <class Builder, class Decoder>
class MultiSource_LinearEncoder : public DialogueBuilder<Builder, Decoder>{
	public:
	using DialogueBuilder<Builder, Decoder>::layers;
    using DialogueBuilder<Builder, Decoder>::hidden_dim;
    using DialogueBuilder<Builder, Decoder>::rep_hidden;
    using DialogueBuilder<Builder, Decoder>::i_U;
    using DialogueBuilder<Builder, Decoder>::p_U;
    using DialogueBuilder<Builder, Decoder>::i_bias;
    using DialogueBuilder<Builder, Decoder>::i_R;
    using DialogueBuilder<Builder, Decoder>::v_decoder;
    using DialogueBuilder<Builder, Decoder>::turnid;
    using DialogueBuilder<Builder, Decoder>::p_cs;
    using DialogueBuilder<Builder, Decoder>::v_src;
    using DialogueBuilder<Builder, Decoder>::src_len;
    using DialogueBuilder<Builder, Decoder>::zero;
    using DialogueBuilder<Builder, Decoder>::src;
    using DialogueBuilder<Builder, Decoder>::save_context;
    using DialogueBuilder<Builder, Decoder>::decoder_single_instance_step;
    using DialogueBuilder<Builder, Decoder>::serialise_cxt_to_external_memory;
    using DialogueBuilder<Builder, Decoder>::vocab_size_tgt;
	using DialogueBuilder<Builder, Decoder>::nutt;
	using DialogueBuilder<Builder, Decoder>::i_h0;
	using DialogueBuilder<Builder, Decoder>::p_h0;
	using DialogueBuilder<Builder, Decoder>::last_context_exp;
	using DialogueBuilder<Builder, Decoder>::p_R;
	using DialogueBuilder<Builder, Decoder>::encoder_bwd;
	using DialogueBuilder<Builder, Decoder>::src_fwd;
	using DialogueBuilder<Builder, Decoder>::src_words;
	using DialogueBuilder<Builder, Decoder>::slen;
	using DialogueBuilder<Builder, Decoder>::decoder;
	using DialogueBuilder<Builder, Decoder>::v_decoder_context;
	using DialogueBuilder<Builder, Decoder>::vocab_size;
	using DialogueBuilder<Builder, Decoder>::tgt_words;
	
	using DialogueBuilder<Builder, Decoder>::p_bias;	
	using DialogueBuilder<Builder, Decoder>::encoder_fwd;
	using DialogueBuilder<Builder, Decoder>::last_decoder_s;
	
	using DialogueBuilder<Builder, Decoder>::v_errs;
    using DialogueBuilder<Builder, Decoder>::serialise;

protected:
    /// time-dependent embedding weight
    map<size_t, map<size_t, tExpression>> m_time_embedding_weight;
    Expression i_zero;
    Builder combiner; /// the combiner that combines the multipe sources of inputs, and possibly its history

    Parameters * p_cxt_to_decoder, *p_enc_to_intention;
    Expression i_cxt_to_decoder, i_enc_to_intention;

    /// for beam search results
    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> completed;

public:
    MultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
			      const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :DialogueBuilder<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale),
        combiner(layers[INTENTION_LAYER], vector<unsigned>{hidden_dims[INTENTION_LAYER], hidden_dims[INTENTION_LAYER], hidden_dims[INTENTION_LAYER]}, &model, iscale)
    {
        p_cxt_to_decoder = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_dim[INTENTION_LAYER] }, iscale, "p_cxt_to_decoder");

        p_enc_to_intention= model.add_parameters({ hidden_dim[INTENTION_LAYER], hidden_dim[ENCODER_LAYER]}, iscale, "p_enc_to_intention");
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg) 
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::start_new_instance" << endl;
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;
        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);
            if (verbose)
                display_value(i_zero,  cg, "i_zero");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            if (verbose)
                cout << "has prv_response" << endl;
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
            if (verbose)
                cout << "done encode prv_response" << endl;
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }
        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));
        if (verbose)
            cout << "done concatenate_cols(average_embedding(slen, source, cg, p_cs))" << endl;

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);
        if (verbose)
            cout << "done encode current input" << endl;

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);
        if (verbose)
            cout << "done combiner add_input" << endl;

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
        if (verbose)
            cout << "done decoder.start_new_sequence" << endl;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::decoder_step" << endl;
        Expression i_c_t;
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
            {
                if (verbose)
                {
                    cout << "to call lookup of " << p << endl;
                }
                i_x_x = lookup(cg, p_cs, p);
                if (verbose)
                    cout << "done call lookup " << p << endl;
            }
            else
            {
                if (verbose)
                    cout << "to call i_zero" << endl;
                i_x_x = i_zero;
            }
            if (verbose)
                cout << " before calling display_value i_x_x" << endl;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        if (verbose)
            cout << "to call concatenate_cols" << endl;
        Expression i_obs = concatenate_cols(v_x_t);
        if (verbose)
            display_value(i_obs, cg, "i_obs");

        if (verbose)
            cout << "to call decoder.add_input" << endl;
        i_h_t = decoder.add_input(i_obs);
        if (verbose)
            display_value(i_h_t, cg, "i_h_t");

        return i_h_t;
    }

    vector<Expression> build_graph(
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        vector<vector<int>> prv_response;
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs;
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            if (verbose)
                display_value(i_y_t, cg, "i_y_t");
            Expression i_r_t = i_R * i_y_t;
            if (verbose)
            {
                cg.incremental_forward(); 
                cout << "i_r_t" << endl;
            }

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            if (verbose)
            {
                cg.incremental_forward();
                cout << "x_r_t" << endl;
            }
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "r_r_t" << endl;
                    }
                    Expression i_ydist = log_softmax(r_r_t);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "i_ydist" << endl;
                    }
                    this_errs[i].push_back(-pick(i_ydist, target_response[i][t + 1]));
                    if (verbose)
                        display_value(this_errs[i].back(), cg, "this_errs");
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : v_decoder.back()->final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input, 
        const std::vector<std::vector<int>>& target_response, 
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            if (verbose)
                display_value(i_y_t, cg, "i_y_t");
            Expression i_r_t = i_R * i_y_t;
            if (verbose)
            {
                cg.incremental_forward();
                cout << "i_r_t" << endl;
            }

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            if (verbose)
            {
                cg.incremental_forward();
                cout << "x_r_t" << endl;
            }
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "r_r_t" << endl;
                    }
                    Expression i_ydist = log_softmax(r_r_t);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "i_ydist" << endl;
                    }
                    this_errs[i].push_back(-pick(i_ydist, target_response[i][t + 1]));
                    if (verbose)
                        display_value(this_errs[i].back(), cg, "this_errs utt[" + boost::lexical_cast<string>(i) + "]");
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "before looping decodre.final_s()" << endl;
                    }
                    
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    if (verbose)
                        display_value(v_t.back(), cg, "v_t");
                    v_decoder_context[i] = v_t;
                    if (verbose)
                    {
                        cout << "v_t pushed" << endl;
                    }
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);
        if (verbose)
            display_value(i_nerr, cg, "i_nerr");
        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    virtual void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response; 

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    /**
    1) save context hidden state
    in last_cxt_s as [replicate_hidden_layers][nutt]
    2) organize the context from decoder
    data is organized in v_decoder_context as [nutt][replicate_hidden_layers]
    after this process, last_decoder_s will save data in dimension [replicate_hidden_layers][nutt]
    */
    void serialise_context(ComputationGraph& cg)
    {
        /// get the top output
        serialise(cg, combiner);

        if (v_decoder_context.size() == 0)
            return;

        vector<vector<cnn::real>> v_last_d;
        unsigned int nutt = v_decoder_context.size();
        size_t ndim = v_decoder_context[0].size();
        v_last_d.resize(ndim);

        size_t ik = 0;
        vector<vector<cnn::real>> vm;
        for (const auto &p : v_decoder_context)
        {
            /// for each utt
            vm.clear();
            for (const auto &v : p)
                vm.push_back(get_value(v, cg));

            size_t iv = 0;
            for (auto p : vm)
            {
                if (ik == 0)
                {
                    v_last_d[iv].resize(nutt * p.size());
                }
                std::copy_n(p.begin(), p.size(), &v_last_d[iv][ik * p.size()]);
                iv++;
            }
            ik++;
        }
        last_decoder_s = v_last_d;
    }

    /// serialise the context to external memory in CPU
    void serialise_cxt_to_external_memory(ComputationGraph& cg, vector<vector<cnn::real>>& ext_memory)
    {
        serialise_cxt_to_external_memory(cg, combiner, ext_memory);
    }

public:
    /// for beam search decoder

    virtual std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {

        //assert(!giza_extensions);
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        size_t tgt_len = 2 * source.size();
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        completed = priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis>();  /// reset the queue
        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
        chart.push(Hypothesis(decoder.state(), sos_sym, 0.0f, 0));

        boost::integer_range<int> vocab = boost::irange<int>(0, vocab_size_tgt);
        vector<int> vec_vocab(vocab_size_tgt, 0);
        for (auto k : vocab)
        {
            vec_vocab[k] = k;
        }
        vector<int> org_vec_vocab = vec_vocab;

        size_t it = 0;
        while (it < tgt_len) {
            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> new_chart;
            vec_vocab = org_vec_vocab;
            real best_score_in_beam = -numeric_limits<real>::infinity() + 100.;

            while (!chart.empty()) {
                Hypothesis hprev = chart.top();
                //Expression i_scores = add_input(hprev.target.back(), hprev.t, cg, &hprev.builder_state);
                Expression i_scores = decoder_single_instance_step(hprev.target.back(), cg, &hprev.builder_state);
                Expression ydist = softmax(i_scores); // compiler warning, but see below

                // find the top k best next words
                //auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
                auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
                real mscore = log(*max_element(dist.begin(), dist.end())) + hprev.cost;
                //if (mscore < best_score - beam_width)
                if (mscore < best_score_in_beam)
                {
                    chart.pop();
                    continue;
                }

                //best_score_in_beam = max(mscore, best_score_in_beam);

                // add to chart
                /*  size_t k = 0;
                for (auto vi : vec_vocab){
                real score = hprev.cost + log(dist[vi]);
                if (score >= best_score - beam_width)
                {
                Hypothesis hnew(combiner.state(), vi, score, hprev);
                if (vi == eos_sym)
                completed.push(hnew);
                else
                new_chart.push(hnew);
                }
                }*/

                vector<real> scores(vocab_size, 0);
                for (auto vi : vec_vocab){
                    scores[vi] = hprev.cost + log(dist[vi]);
                    //real score = hprev.cost + log(dist[vi]);
                }

                std::sort(scores.begin(), scores.end(), greater<real>());
                best_score_in_beam = scores[beam_width - 1];

                for (auto vi : vec_vocab)
                {
                    real score = hprev.cost + log(dist[vi]);
                    if (score > best_score_in_beam)
                    {
                        if (vi == eos_sym)
                        {
                            Hypothesis hnew(decoder.state(), vi / (it+1), score, hprev);
                            completed.push(hnew);
                        }
                        else
                        {
                            Hypothesis hnew(decoder.state(), vi, score, hprev);
                            new_chart.push(hnew);
                        }
                    }
                }

                chart.pop();
            }

            if (new_chart.size() == 0)
                break;

            size_t top_beam_width = 0;
            while (!new_chart.empty() && top_beam_width < beam_width)
            {
                /*if (new_chart.top().cost > best_score - beam_width){
                chart.push(new_chart.top());
                }
                else
                break;
                new_chart.pop();*/
                chart.push(new_chart.top());
                new_chart.pop();
                top_beam_width++;
            }
            it++;
        }

        vector<int> best;
        if (completed.size() == 0)
        {
            cerr << "beam search decoding beam width too small, use the best path so far" << flush;

            best = chart.top().target;
            best.push_back(eos_sym);
        }
        else
        {
            best = completed.top().target;
        }


        save_context(cg);
        serialise_context(cg);
        turnid++;
        return best;
    }

    virtual std::vector<int> beam_decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {
        //assert(!giza_extensions);
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        size_t tgt_len = 2 * source.size();

        start_new_single_instance(prv_response, source, cg);

        completed = priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis>();  /// reset the queue
        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
        chart.push(Hypothesis(decoder.state(), sos_sym, 0.0f, 0));

        boost::integer_range<int> vocab = boost::irange<int>(0, vocab_size_tgt);
        vector<int> vec_vocab(vocab_size_tgt, 0);
        for (auto k : vocab)
        {
            vec_vocab[k] = k;
        }
        vector<int> org_vec_vocab = vec_vocab;

        size_t it = 0;
        while (it < tgt_len) {
            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> new_chart;
            vec_vocab = org_vec_vocab;
            real best_score_in_beam = -numeric_limits<real>::infinity() + 100.;

            while (!chart.empty()) {
                Hypothesis hprev = chart.top();
                //Expression i_scores = add_input(hprev.target.back(), hprev.t, cg, &hprev.builder_state);
                Expression i_scores = decoder_single_instance_step(hprev.target.back(), cg, &hprev.builder_state);
                Expression ydist = softmax(i_scores); // compiler warning, but see below

                // find the top k best next words
                //auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
                auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
                real mscore = log(*max_element(dist.begin(), dist.end())) + hprev.cost;
                //if (mscore < best_score - beam_width)
                if (mscore < best_score_in_beam)
                {
                    chart.pop();
                    continue;
                }

                //best_score_in_beam = max(mscore, best_score_in_beam);

                // add to chart
                /*  size_t k = 0;
                for (auto vi : vec_vocab){
                real score = hprev.cost + log(dist[vi]);
                if (score >= best_score - beam_width)
                {
                Hypothesis hnew(combiner.state(), vi, score, hprev);
                if (vi == eos_sym)
                completed.push(hnew);
                else
                new_chart.push(hnew);
                }
                }*/

                vector<real> scores(vocab_size, 0);
                for (auto vi : vec_vocab){
                    scores[vi] = hprev.cost + log(dist[vi]);
                    //real score = hprev.cost + log(dist[vi]);
                }

                std::sort(scores.begin(), scores.end(), greater<real>());
                best_score_in_beam = scores[beam_width - 1];

                for (auto vi : vec_vocab)
                {
                    real score = hprev.cost + log(dist[vi]);
                    if (score > best_score_in_beam)
                    {
                        if (vi == eos_sym)
                        {
                            Hypothesis hnew(decoder.state(), vi / (it+1), score, hprev);
                            completed.push(hnew);
                        }
                        else
                        {
                            Hypothesis hnew(decoder.state(), vi, score, hprev);
                            new_chart.push(hnew);
                        }
                    }
                }

                chart.pop();
            }

            if (new_chart.size() == 0)
                break;

            size_t top_beam_width = 0;
            while (!new_chart.empty() && top_beam_width < beam_width)
            {
                /*if (new_chart.top().cost > best_score - beam_width){
                chart.push(new_chart.top());
                }
                else
                break;
                new_chart.pop();*/
                chart.push(new_chart.top());
                new_chart.pop();
                top_beam_width++;
            }
            it++;
        }

        vector<int> best;
        if (completed.size() == 0)
        {
            cerr << "beam search decoding beam width too small, use the best path so far" << flush;

            best = chart.top().target;
            best.push_back(eos_sym);
        }
        else
        {
            best = completed.top().target;
        }

        save_context(cg);
        serialise_context(cg);
        turnid++;
        return best;
    }

public:
    /// for reranking
    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> get_beam_decode_complete_list()
    {
        return completed;
    }

};

/** additionally with attention
  the decoder response from the last turn is used as input to a RNN, 
  the last state of this RNN is the initial state of another RNN, which takes input from the current user input
  the last state of the another RNN is used as input to the intention RNN,
  the last state of the intention RNN is the initial state of the decoder RNN,
  the decoder RNN uses attention to the current user input
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder : public MultiSource_LinearEncoder <Builder, Decoder>{
	public:
	using DialogueBuilder<Builder, Decoder>::layers;
    using DialogueBuilder<Builder, Decoder>::hidden_dim;
    using DialogueBuilder<Builder, Decoder>::rep_hidden;
    using DialogueBuilder<Builder, Decoder>::i_U;
    using DialogueBuilder<Builder, Decoder>::p_U;
    using DialogueBuilder<Builder, Decoder>::i_bias;
    using DialogueBuilder<Builder, Decoder>::i_R;
    using DialogueBuilder<Builder, Decoder>::v_decoder;
    using DialogueBuilder<Builder, Decoder>::turnid;
    using DialogueBuilder<Builder, Decoder>::p_cs;
    using DialogueBuilder<Builder, Decoder>::v_src;
    using DialogueBuilder<Builder, Decoder>::src_len;
    using DialogueBuilder<Builder, Decoder>::zero;
    using DialogueBuilder<Builder, Decoder>::src;
    using DialogueBuilder<Builder, Decoder>::save_context;
    using DialogueBuilder<Builder, Decoder>::decoder_single_instance_step;
	
	using DialogueBuilder<Builder, Decoder>::nutt;
	using DialogueBuilder<Builder, Decoder>::i_h0;
	using DialogueBuilder<Builder, Decoder>::p_h0;
	using DialogueBuilder<Builder, Decoder>::last_context_exp;
	using DialogueBuilder<Builder, Decoder>::p_R;
	using DialogueBuilder<Builder, Decoder>::encoder_bwd;
	using DialogueBuilder<Builder, Decoder>::src_fwd;
	using DialogueBuilder<Builder, Decoder>::src_words;
	using DialogueBuilder<Builder, Decoder>::slen;
	using DialogueBuilder<Builder, Decoder>::decoder;
	using DialogueBuilder<Builder, Decoder>::v_decoder_context;
	using DialogueBuilder<Builder, Decoder>::vocab_size;
	using DialogueBuilder<Builder, Decoder>::tgt_words;
	
	using DialogueBuilder<Builder, Decoder>::p_bias;	
	using DialogueBuilder<Builder, Decoder>::encoder_fwd;
	using DialogueBuilder<Builder, Decoder>::last_decoder_s;
	
	using MultiSource_LinearEncoder<Builder, Decoder>::combiner;
		
	using DialogueBuilder<Builder, Decoder>::v_errs;
	using MultiSource_LinearEncoder<Builder, Decoder>::i_cxt_to_decoder;
	using MultiSource_LinearEncoder<Builder, Decoder>::p_cxt_to_decoder;
	using MultiSource_LinearEncoder<Builder, Decoder>::i_enc_to_intention;
	using MultiSource_LinearEncoder<Builder, Decoder>::p_enc_to_intention;
	
    using MultiSource_LinearEncoder<Builder, Decoder>::serialise_context;
    using MultiSource_LinearEncoder<Builder, Decoder>::beam_decode;

protected:
    /// for location prediction for local attention
    Parameters* p_va_local, *p_Wa_local, *p_ba_local;
    Expression i_va_local, i_Wa_local, i_ba_local;

    /// for the expoential scaling of the softmax
    Parameters* p_scale;
    Expression i_scale;

    Parameters* p_Wa, *p_va;
    Expression i_Wa, i_va;

    /// a single layer MLP
    DNNBuilder attention_layer;
    vector<Expression> attention_output_for_this_turn; /// [number of turn]

    Expression i_zero;
public:
    AttMultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
				 const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :MultiSource_LinearEncoder<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale) ,
        attention_layer(1, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER]}, &model, iscale, "attention_layer")
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder" << endl;

        if (hidden_dim[ENCODER_LAYER] != hidden_dim[EMBEDDING_LAYER] ||
            hidden_dim[DECODER_LAYER] != hidden_dim[ENCODER_LAYER])
        {
            cout << "Warning: not the same dimension for encoder, decoder and embedding" << endl;
            cout << "Warning: might be okay for derived class but if just using AttMultiSource_LinearEncoder can trigger runtime error later" << endl;
        }

        p_Wa_local = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[DECODER_LAYER] }, iscale);
        p_ba_local = model.add_parameters({ hidden_dim[ALIGN_LAYER] }, iscale);
        p_va_local = model.add_parameters({ 1, hidden_dim[ALIGN_LAYER] }, iscale);

        unsigned int align_dim = hidden_dim[ALIGN_LAYER];
        p_Wa = model.add_parameters({ align_dim, hidden_dim[DECODER_LAYER] }, iscale);
        p_va = model.add_parameters({ align_dim }, iscale);

        p_scale = model.add_parameters({ 1 }, 0.0);

    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg) 
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_output_for_this_turn.clear();

            i_Wa_local = parameter(cg, p_Wa_local);
            i_va_local = parameter(cg, p_va_local);
            i_ba_local = parameter(cg, p_ba_local);

            i_scale = exp(parameter(cg, p_scale));

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        Expression src_tmp = concatenate_cols(embedding(slen, source, cg, p_cs, zero, hidden_dim[ENCODER_LAYER]));
        v_src = shuffle_data(src_tmp, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        if (verbose)
            display_value(i_input, cg, "i_input = ");
        i_h_t = decoder.add_input(i_input);
        if (verbose)
            display_value(i_h_t, cg, "i_h_t = ");

        vector<Expression> alpha;
        vector<Expression> position = local_attention_to(cg, src_len, i_Wa_local, i_ba_local, i_va_local, i_h_t, nutt);
        if (verbose)
        {
            for (auto &p : position){
                display_value(p, cg, "predicted local attention position ");
            }
        }

        vector<Expression> v_context_to_source = attention_using_bilinear_with_local_attention(v_src, src_len, i_Wa, i_h_t, hidden_dim[ALIGN_LAYER], nutt, alpha, i_scale, position);
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute attention
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        if (verbose) display_value(i_combined_input_to_attention, cg, "i_combined_input_to_attention = ");
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);
        if (verbose) display_value(i_h_attention_t, cg, "i_h_attention_t  = ");

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
    
    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;
            Expression i_ydist = log_softmax(i_r_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size; 
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;
            Expression i_ydist = log_softmax(i_r_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size; 
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

};

/**
Use attentional max-entropy features or direct features
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithMaxEntropyFeature : public AttMultiSource_LinearEncoder <Builder, Decoder>{
	public:
	using DialogueBuilder<Builder, Decoder>::layers;
    using DialogueBuilder<Builder, Decoder>::hidden_dim;
    using DialogueBuilder<Builder, Decoder>::rep_hidden;
    using DialogueBuilder<Builder, Decoder>::i_U;
    using DialogueBuilder<Builder, Decoder>::p_U;
    using DialogueBuilder<Builder, Decoder>::i_bias;
    using DialogueBuilder<Builder, Decoder>::i_R;
    using DialogueBuilder<Builder, Decoder>::v_decoder;
    using DialogueBuilder<Builder, Decoder>::turnid;
    using DialogueBuilder<Builder, Decoder>::p_cs;
    using DialogueBuilder<Builder, Decoder>::v_src;
    using DialogueBuilder<Builder, Decoder>::src_len;
    using DialogueBuilder<Builder, Decoder>::zero;
    using DialogueBuilder<Builder, Decoder>::src;
    using DialogueBuilder<Builder, Decoder>::save_context;
    using DialogueBuilder<Builder, Decoder>::assign_cxt;
    using DialogueBuilder<Builder, Decoder>::copy_external_memory_to_cxt;
    using DialogueBuilder<Builder, Decoder>::decoder_single_instance_step;
	
	using DialogueBuilder<Builder, Decoder>::nutt;
	using DialogueBuilder<Builder, Decoder>::i_h0;
	using DialogueBuilder<Builder, Decoder>::p_h0;
	using DialogueBuilder<Builder, Decoder>::last_context_exp;
	using DialogueBuilder<Builder, Decoder>::p_R;
	using DialogueBuilder<Builder, Decoder>::encoder_bwd;
	using DialogueBuilder<Builder, Decoder>::src_fwd;
	using DialogueBuilder<Builder, Decoder>::src_words;
	using DialogueBuilder<Builder, Decoder>::slen;
	using DialogueBuilder<Builder, Decoder>::decoder;
	using DialogueBuilder<Builder, Decoder>::v_decoder_context;
	using DialogueBuilder<Builder, Decoder>::vocab_size;
	using DialogueBuilder<Builder, Decoder>::tgt_words;
	
	using DialogueBuilder<Builder, Decoder>::p_bias;	
	using DialogueBuilder<Builder, Decoder>::encoder_fwd;
	using DialogueBuilder<Builder, Decoder>::last_decoder_s;
	
	using MultiSource_LinearEncoder<Builder, Decoder>::combiner;
		
	using DialogueBuilder<Builder, Decoder>::v_errs;
	using MultiSource_LinearEncoder<Builder, Decoder>::i_cxt_to_decoder;
	using MultiSource_LinearEncoder<Builder, Decoder>::p_cxt_to_decoder;
	using MultiSource_LinearEncoder<Builder, Decoder>::i_enc_to_intention;
	using MultiSource_LinearEncoder<Builder, Decoder>::p_enc_to_intention;
	
	using AttMultiSource_LinearEncoder<Builder, Decoder>::i_Wa;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::p_Wa;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::i_va;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::p_va;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::attention_layer;
	using DialogueBuilder<Builder, Decoder>::vocab_size_tgt;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::i_zero;
	using AttMultiSource_LinearEncoder<Builder, Decoder>::attention_output_for_this_turn;
    using MultiSource_LinearEncoder<Builder, Decoder>::beam_decode;
    using MultiSource_LinearEncoder<Builder, Decoder>::serialise_cxt_to_external_memory;
    using MultiSource_LinearEncoder<Builder, Decoder>::serialise_context;

    using MultiSource_LinearEncoder<Builder, Decoder>::completed;
    using MultiSource_LinearEncoder<Builder, Decoder>::get_beam_decode_complete_list;

protected:
    cnn::real r_softmax_scale; /// for attention softmax exponential scale
    LookupParameters* p_max_ent; /// weight for max-entropy feature

    vector<Expression> v_max_ent_obs; /// observation from max-ent feature
    Expression        i_max_ent_obs;

    Parameters * p_emb2enc; /// embedding to encoding
    Parameters * p_emb2enc_b; /// bias 
    Expression   i_emb2enc;
    Expression   i_emb2enc_b;           /// the bias that is to be applied to each sentence
    Expression   i_emb2enc_b_all_words; /// the bias that is to be applied to every word

    Expression i_zero_emb; /// Expresison for embedding of zeros in the embedding space
    vector<cnn::real> zero_emb; 

public:
    AttMultiSource_LinearEncoder_WithMaxEntropyFeature(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
						       const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature" << endl;

        r_softmax_scale = 1.0;

        unsigned int align_dim = hidden_dim[ALIGN_LAYER];
        if (p_Wa == nullptr) p_Wa = model.add_parameters({ align_dim, hidden_dim[DECODER_LAYER] }, iscale, "p_Wa");
        if (p_va == nullptr) p_va = model.add_parameters({ align_dim }, iscale, "p_va");
        p_U  = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[ENCODER_LAYER] }, iscale, "p_U");

        /// bi-gram weight
        p_max_ent = model.add_lookup_parameters(vocab_size_tgt, { vocab_size_tgt }, iscale);

        /// embedding to encoding
        p_emb2enc = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] });
        p_emb2enc_b = model.add_parameters({ hidden_dim[ENCODER_LAYER] });

        zero_emb.resize(hidden_dim[EMBEDDING_LAYER], 0.0);
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_bias)));

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);
            i_zero_emb = input(cg, { (hidden_dim[EMBEDDING_LAYER]) }, &zero_emb);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);

            i_emb2enc = parameter(cg, p_emb2enc);
            i_emb2enc_b = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_emb2enc_b)));
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(i_emb2enc * prv_response_enc + i_emb2enc_b);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        int nwords = 0; 
        for (auto p : src_len)
        {
            src_words += (p - 1);
            nwords += p;
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs); 
        i_emb2enc_b_all_words = concatenate_cols(vector<Expression>(nwords, parameter(cg, p_emb2enc_b)));
        src = i_U * (i_emb2enc * concatenate_cols(v_src) + i_emb2enc_b_all_words);

        /// get the representation of inputs for max-entropy feature
        v_max_ent_obs = embedding_spkfirst(source, cg, p_max_ent);
        i_max_ent_obs = concatenate_cols(average_embedding(src_len, vocab_size_tgt, v_max_ent_obs));

        /// get the raw encodeing from source
        src_fwd = i_emb2enc * concatenate_cols(average_embedding(src_len, hidden_dim[EMBEDDING_LAYER], v_src)) + i_emb2enc_b;

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

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

        Expression i_obs = i_emb2enc * concatenate_cols(v_x_t) + i_emb2enc_b;
        Expression i_input;
 
        if (prev_state == nullptr)
        {
            if (attention_output_for_this_turn.size() <= turnid)
            {
                i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
            }
            else
            {
                i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
            }
        }
        else{
            if (*prev_state < 0)
            {
                i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
            }
            else
            {
                i_input = concatenate({ i_obs, attention_output_for_this_turn[*prev_state] });
            }
        }

        i_h_t = decoder.add_input(*prev_state, i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = i_emb2enc * concatenate_cols(v_context_to_source) + i_emb2enc_b;
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        attention_output_for_this_turn.push_back(i_h_attention_t);
        
        Expression i_output = i_R * i_h_attention_t;
        Expression i_comb_max_entropy = i_output + i_max_ent_obs;

        return i_comb_max_entropy + i_bias;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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

        Expression i_obs = i_emb2enc * concatenate_cols(v_x_t) + i_emb2enc_b;
        Expression i_input;
        if (attention_output_for_this_turn.size() == 0)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = i_emb2enc * concatenate_cols(v_context_to_source) + i_emb2enc_b;
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src});
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        attention_output_for_this_turn.push_back(i_h_attention_t);

        Expression i_output = i_R * i_h_attention_t;
        Expression i_comb_max_entropy = i_output + i_max_ent_obs; 
        
        return i_comb_max_entropy + i_bias;
    }

    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    /// serialise the context to external memory in CPU
    void serialise_cxt_to_external_memory(ComputationGraph& cg, vector<vector<cnn::real>>& ext_memory)
    {
        serialise_cxt_to_external_memory(cg, combiner, ext_memory);
    }

    std::vector<int> decode_with_additional_feature(const std::vector<int> &source, const vector<cnn::real>& additional_feature, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        return vector<int>();
    }

    std::vector<int> sample(const std::vector<int> &prv_context, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        //  std::cerr << tdict.Convert(target.back());
        int t = 0;

        start_new_single_instance(prv_context, source, cg);

        i_bias = parameter(cg, p_bias);
        i_R = parameter(cg, p_R);

        v_decoder_context.clear();
        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_bias + i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
            unsigned w = sample_accoding_to_distribution_of(dist);
            auto pr_w = dist[w];

            // break potential infinite loop
            if (t > 100) {
                w = eos_sym;
                pr_w = dist[w];
            }

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        vector<Expression> v_t = decoder.final_s();

        save_context(cg);

        turnid++;

        return target;
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode_with_additional_feature(const std::vector<int> &prv_response, const std::vector<int> &source, const vector<cnn::real>&, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        return vector<int>();
    }
    
    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &source, const vector<cnn::real>&, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {
        return vector<int>();
    }
    
    virtual std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {

        //assert(!giza_extensions);
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        size_t tgt_len = 40;//50 * source.size();
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        completed = priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis>();  /// reset the queue
        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
        chart.push(Hypothesis(decoder.state(), sos_sym, 0.0f, 0));

        boost::integer_range<int> vocab = boost::irange<int>(0, vocab_size_tgt);
        vector<int> vec_vocab(vocab_size_tgt, 0);
        for (auto k : vocab)
        {
            vec_vocab[k] = k;
        }
        vector<int> org_vec_vocab = vec_vocab;

        size_t it = 0;
        while (it < tgt_len) {
            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> new_chart;
            vec_vocab = org_vec_vocab;
            real best_score = -numeric_limits<real>::infinity() + 100.;

            while (!chart.empty()) {
                Hypothesis hprev = chart.top();
                //Expression i_scores = add_input(hprev.target.back(), hprev.t, cg, &hprev.builder_state);
                Expression i_scores = decoder_single_instance_step(hprev.target.back(), cg, &hprev.builder_state);
                Expression ydist = log_softmax(i_scores); // compiler warning, but see below

                // find the top k best next words
                //auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
                auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
                real mscore = *max_element(dist.begin(), dist.end()) + hprev.cost;
                if (mscore < best_score - beam_width)
                {
                    chart.pop();
                    continue;
                }

                best_score = max(mscore, best_score);

                for (auto vi : vec_vocab)
                {
                    real score = hprev.cost + dist[vi];
                    if (score >= best_score - beam_width)
                    {
                        if (vi == eos_sym)
                        {
                            Hypothesis hnew(decoder.state(), vi, score / (it + 1), hprev);
                            completed.push(hnew);
                        }
                        else
                        {
                            Hypothesis hnew(decoder.state(), vi, score, hprev);
                            new_chart.push(hnew);
                        }
                    }
                }

                chart.pop();
            }

            if (new_chart.size() == 0)
                break;

            while (!new_chart.empty() && chart.size() <= max_number_of_hypothesis)
            {
                if (new_chart.top().cost > best_score - beam_width)
                {
                    chart.push(new_chart.top());
                    new_chart.pop();
                }
                else
                    break;
            }
            it++;
        }

        vector<int> best;
        if (completed.size() == 0)
        {
            cerr << "beam search decoding beam width too small, use the best path so far" << flush;

            best = chart.top().target;
            best.push_back(eos_sym);
        }
        else
        {
            best = completed.top().target;
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return best;
    }

    std::vector<int> beam_decode_with_additional_feature(const std::vector<int> &prv_response, const std::vector<int> &source, const vector<cnn::real>&, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {
        return vector<int>();
    }
    
    virtual std::vector<int> beam_decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
    {
        //assert(!giza_extensions);
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        size_t tgt_len = 30;//50 * source.size();

        start_new_single_instance(prv_response, source, cg);

        completed = priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis>();  /// reset the queue
        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
        chart.push(Hypothesis(decoder.state(), sos_sym, 0.0f, 0));

        boost::integer_range<int> vocab = boost::irange<int>(0, vocab_size_tgt);
        vector<int> vec_vocab(vocab_size_tgt, 0);
        for (auto k : vocab)
        {
            vec_vocab[k] = k;
        }
        vector<int> org_vec_vocab = vec_vocab;

        size_t it = 0;
        while (it < tgt_len) {
            priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> new_chart;
            vec_vocab = org_vec_vocab;
            real best_score = -numeric_limits<real>::infinity() + 100.;

            while (!chart.empty()) {
                Hypothesis hprev = chart.top();
                //Expression i_scores = add_input(hprev.target.back(), hprev.t, cg, &hprev.builder_state);
                Expression i_scores = decoder_single_instance_step(hprev.target.back(), cg, &hprev.builder_state);
                Expression ydist = log_softmax(i_scores); // compiler warning, but see below

                // find the top k best next words
                //auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
                auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
                real mscore = *max_element(dist.begin(), dist.end()) + hprev.cost;
                if (mscore < best_score - beam_width)
                {
                    chart.pop();
                    continue;
                }

                best_score = max(mscore, best_score);

                for (auto vi : vec_vocab)
                {
                    real score = hprev.cost + dist[vi];
                    if (score >= best_score - beam_width)
                    {
                        if (vi == eos_sym)
                        {
                            Hypothesis hnew(decoder.state(), vi, score / (it + 1), hprev);
                            completed.push(hnew);
                        }
                        else
                        {
                            Hypothesis hnew(decoder.state(), vi, score, hprev);
                            new_chart.push(hnew);
                        }
                    }
                }

                chart.pop();
            }

            if (new_chart.size() == 0)
                break;

            while (!new_chart.empty() && chart.size() <= max_number_of_hypothesis)
            {
                if (new_chart.top().cost > best_score - beam_width)
                {
                    chart.push(new_chart.top());
                    new_chart.pop();
                }
                else
                    break;
            }
            it++;
        }

        vector<int> best;
        if (completed.size() == 0)
        {
            cerr << "beam search decoding beam width too small, use the best path so far" << flush;

            best = chart.top().target;
            best.push_back(eos_sym);
        }
        else
        {
            best = completed.top().target;
        }


        save_context(cg);
        serialise_context(cg);

        /// n-best
        /*
        int kbest = 0;
        while (completed.size() != 0)
        {
            auto pbest = completed.top().target;
            cout << "top" << kbest++ << " best : ";
            for (auto a : pbest)
                cout << sd.Convert(a) << " ";
            cout << endl;
            completed.pop();
            if (kbest > 3)
                break;
        }*/

        turnid++;
        return best;
    }

    /// return [nutt][decoded_results]
    std::vector<Sentence> batch_decode(const std::vector<Sentence>& prv_response, 
        const std::vector<Sentence> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        unsigned int nutt = source.size();
        std::vector<Sentence> target(nutt, vector<int>(1, sos_sym));

        int t = 0;

        start_new_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        vector<int> vtmp(nutt, sos_sym);

        bool need_decode = true;
        while (need_decode)
        {
            Expression i_y_t = log_softmax(decoder_step(vtmp, cg));
            Expression ydist = reshape(i_y_t, { nutt * vocab_size_tgt });
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist

            need_decode = false;
            vtmp.clear();
            for (size_t k = 0; k < nutt; k++)
            {
                if (target[k].back() != eos_sym)
                {
                    // find the argmax next word (greedy)
                    unsigned w = 0;
                    unsigned init_pos = k * vocab_size_tgt;
                    auto pr_w = dist[w + init_pos];
                    for (unsigned x = 1 + init_pos; x < init_pos + vocab_size_tgt; ++x) {
                        if (dist[x] > pr_w) {
                            w = x;
                            pr_w = dist[x];
                        }
                    }

                    // break potential infinite loop
                    if (t > 100) {
                        w = eos_sym + init_pos;
                        pr_w = dist[w];
                    }

                    vtmp.push_back(w - init_pos);
                    target[k].push_back(w - init_pos);
                    need_decode = true;
                }
                else
                    vtmp.push_back(eos_sym);
            }
            t += 1;
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;

        return target;
    }

};

/**
Use attentional max-entropy features or direct features
*/
/* template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithHashingMaxEntropyFeature : public AttMultiSource_LinearEncoder <Builder, Decoder>{
protected:
    cnn::real r_softmax_scale; /// for attention softmax exponential scale
    LookupParameters* p_max_ent; /// weight for max-entropy feature

    vector<Expression> v_max_ent_obs; /// observation from max-ent feature
    Expression        i_max_ent_obs;
    int m_hash_size; 
    int m_direct_order; /// max-entropy feature order

    Parameters * p_emb2enc; /// embedding to encoding
    Parameters * p_emb2enc_b; /// bias 
    Expression   i_emb2enc;
    Expression   i_emb2enc_b;           /// the bias that is to be applied to each sentence
    Expression   i_emb2enc_b_all_words; /// the bias that is to be applied to every word

    Expression i_zero_emb; /// Expresison for embedding of zeros in the embedding space
    vector<cnn::real> zero_emb;

public:
    AttMultiSource_LinearEncoder_WithHashingMaxEntropyFeature(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature" << endl;

        r_softmax_scale = 1.0;

        unsigned int align_dim = hidden_dim[ALIGN_LAYER];
        if (p_Wa == nullptr) p_Wa = model.add_parameters({ align_dim, hidden_dim[DECODER_LAYER] }, iscale, "p_Wa");
        if (p_va == nullptr) p_va = model.add_parameters({ align_dim }, iscale, "p_va");
        p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[ENCODER_LAYER] }, iscale, "p_U");

        /// bi-gram weight
        m_hash_size = hidden_dims[HASHING_LAYER];
        m_direct_order = hidden_dims[MEORDER_LAYER];
        p_max_ent = model.add_lookup_parameters(m_hash_size, { vocab_size_tgt }, iscale);

        /// embedding to encoding
        p_emb2enc = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[EMBEDDING_LAYER] });
        p_emb2enc_b = model.add_parameters({ hidden_dim[ENCODER_LAYER] });

        zero_emb.resize(hidden_dim[EMBEDDING_LAYER], 0.0);
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_bias)));

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);
            i_zero_emb = input(cg, { (hidden_dim[EMBEDDING_LAYER]) }, &zero_emb);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);

            i_emb2enc = parameter(cg, p_emb2enc);
            i_emb2enc_b = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_emb2enc_b)));
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(i_emb2enc * prv_response_enc + i_emb2enc_b);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        int nwords = 0;
        for (auto p : src_len)
        {
            src_words += (p - 1);
            nwords += p;
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs);
        i_emb2enc_b_all_words = concatenate_cols(vector<Expression>(nwords, parameter(cg, p_emb2enc_b)));
        src = i_U * (i_emb2enc * concatenate_cols(v_src) + i_emb2enc_b_all_words);

        /// get the representation of inputs
        v_max_ent_obs = hash_embedding_spkfirst(source, cg, p_max_ent, m_direct_order, m_hash_size);
        /// expand to accomodate order
        vector<unsigned> hash_len;
        for (auto p : src_len)
            hash_len.push_back(p * m_direct_order);
        i_max_ent_obs = concatenate_cols(average_embedding(hash_len, vocab_size_tgt, v_max_ent_obs));

        /// get the raw encodeing from source
        src_fwd = i_emb2enc * concatenate_cols(average_embedding(src_len, hidden_dim[EMBEDDING_LAYER], v_src)) + i_emb2enc_b;

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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

        Expression i_obs = i_emb2enc * concatenate_cols(v_x_t) + i_emb2enc_b;
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = i_emb2enc * concatenate_cols(v_context_to_source) + i_emb2enc_b;
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        Expression i_output = i_R * i_h_attention_t;
        Expression i_comb_max_entropy = i_output + i_max_ent_obs;

        return i_comb_max_entropy + i_bias;
    }

    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }
};
 */
/**
Use global max-entropy and attentional direct features
To speed-up, each source has been padded with </s> to have equal length of source sentences
*/
/* template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature_Batch : public AttMultiSource_LinearEncoder_WithMaxEntropyFeature<Builder, Decoder>{
protected:
    LookupParameters* p_tgt_side_emb;  /// target side embedding
    Parameters * p_global_me_weight;/// weight for global ME feature
    Expression i_global_me_weight;

    Expression i_glb_me_feature;     /// computed global me feature

public:
    AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature_Batch(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder_WithMaxEntropyFeature(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature" << endl;

        /// have bigram history to have trigram ME feature
        p_tgt_side_emb = model.add_lookup_parameters(vocab_size_src, { hidden_dim[DECODER_LAYER] }, iscale);
        p_global_me_weight = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            i_global_me_weight = parameter(cg, p_global_me_weight);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs);
        src = i_U * concatenate_cols(v_src);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(src_len, hidden_dim[ENCODER_LAYER], v_src));

        ////// get the global ME feature  //////
        /// first get the representation of inputs
        vector<Expression> v_target_side_emb = embedding_spkfirst(source, cg, p_tgt_side_emb);
        vector<Expression> v_ave_global_me_raw = average_embedding(src_len, hidden_dim[DECODER_LAYER], v_target_side_emb);
        i_glb_me_feature = i_global_me_weight * concatenate_cols(v_ave_global_me_raw);

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source_batch(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = concatenate_cols(v_context_to_source);
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        Expression i_output = i_R * i_h_attention_t
            + concatenated_src
            + i_glb_me_feature;

        return i_output;
    }
    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

}; */

/* template <class Builder, class Decoder>
class ClsBasedMultiSource_LinearEncoder : public MultiSource_LinearEncoder <Builder, Decoder>{
protected:
    ClsBasedBuilder* p_clsbased_error;
    cnn::real iscale; 
    vector<int> hash; 

public:
    ClsBasedMultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :MultiSource_LinearEncoder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale) , iscale(iscale)
    {
        p_clsbased_error = nullptr;
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            p_clsbased_error->new_graph(cg);
            p_clsbased_error->set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }
        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
    }

    bool load_cls_info_from_file(string word2clsfn, string clsszefn, Dict& sd, Model& model)
    {
        vector<int> clssize;
        vector<long> word2cls;
        vector<long> dict_wrd_id2within_class_id;

        ClsBasedBuilder * p_tmp_clsbased  = new ClsBasedBuilder();
        p_tmp_clsbased->load_word2cls_fn(word2clsfn, sd, word2cls, dict_wrd_id2within_class_id, clssize);
        delete p_tmp_clsbased;

        p_clsbased_error = new ClsBasedBuilder(hidden_dim[DECODER_LAYER], clssize, word2cls, dict_wrd_id2within_class_id, model, iscale, "class-based scoring");
        return true;
    }

    ~ClsBasedMultiSource_LinearEncoder()
    {
        delete[] p_clsbased_error;
    }

    vector<Expression> build_graph(
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        unsigned int nutt = current_user_input.size();
        vector<vector<int>> prv_response;
        start_new_instance(prv_response, current_user_input, cg);

        vector<Expression> errs;

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            vector<long> vtarget;
            for (auto p : target_response)
            {
                if (t + 1< p.size())
                    vtarget.push_back(p[t + 1]);
                else
                    vtarget.push_back(-1);
            }
            errs.push_back(sum(p_clsbased_error->add_input(i_y_t, vtarget)));

            for (size_t i = 0; i < nutt; i++)
            {
                if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<Expression> errs;

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            vector<long> vtarget;
            for (auto p : target_response)
            {
                if (t + 1< p.size())
                    vtarget.push_back(p[t + 1]);
                else
                    vtarget.push_back(-1);
            }
            vector<Expression> i_r_t = p_clsbased_error->add_input(i_y_t, vtarget);
            if (i_r_t.size() > 0)
                errs.push_back(sum(i_r_t));
            tgt_words += i_r_t.size();

            for (size_t i = 0; i < nutt; i++)
            {
                if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = p_clsbased_error->respond(i_y_t, cg); 
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = p_clsbased_error->respond(i_y_t, cg);
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

}; */


}; // namespace cnn
