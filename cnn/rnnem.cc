#include "cnn/rnnem.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;


namespace cnn {

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

    enum {WK, WKB, EXTMEM, WBETA, WG };

    RNNEMBuilder::RNNEMBuilder(long layers,
        long input_dim,
        long hidden_dim,
        Model* model) : 
        layers(layers), m_mem_dim(hidden_dim)
    {
        m_mem_size = RNNEM_MEM_SIZE;

        ram = GRUBuilder(layers, input_dim + m_mem_dim, hidden_dim, model);

        unsigned mem_dim = m_mem_dim;
        long mem_size = m_mem_size;
        unsigned layer_input_dim = input_dim;
        for (unsigned i = 0; i < layers; ++i) {
            // for key generation
            Parameters* p_wk = model->add_parameters({ (long)mem_dim, (long)layer_input_dim });
            Parameters* p_wkb = model->add_parameters({ (long)mem_dim });

            // for memory at this layer
            Parameters* p_mem = model->add_parameters({ (long)mem_dim, (long)mem_size });

            // for scaling
            Parameters* p_beta = model->add_parameters({ (long)1, (long)layer_input_dim });

            /// for interploation
            Parameters* p_interpolation = model->add_parameters({ (long)1, (long)layer_input_dim });

            vector<Parameters*> ps = {p_wk, p_wkb, p_mem, p_beta, p_interpolation };

            layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

            params.push_back(ps);
        }  // layers
    }

    void RNNEMBuilder::new_graph_impl(ComputationGraph& cg){
        param_vars.clear();

        for (unsigned i = 0; i < layers; ++i){
            auto& p = params[i];

            // key
            Expression i_wk = parameter(cg, p[WK]);
            Expression i_wkb = parameter(cg, p[WKB]);

            // memory 
            Expression i_mem = parameter(cg, p[EXTMEM]);

            // memory 
            Expression i_beta = parameter(cg, p[WBETA]);

            // memory 
            Expression i_interpolation = parameter(cg, p[WG]);

            vector<Expression> vars = {i_wk, i_wkb, i_mem, i_beta, i_interpolation };
            param_vars.push_back(vars);
        }

        ram.new_graph_impl(cg);
    }

    void RNNEMBuilder::start_new_sequence_impl(const vector<Expression>& hinit)
    {
        w.clear();

        if (hinit.size() > 0) {
            assert(layers * 2 == hinit.size());
            w0.resize(layers);

            vector<Expression> h0;
            for (unsigned i = 0; i < layers; ++i) {
                w0[i] = hinit[i];
                h0.push_back( hinit[i + layers] );
            }

            has_initial_state = true;
            ram.start_new_sequence_impl(h0);
        }
        else {
            has_initial_state = false;
            ram.start_new_sequence_impl(hinit);
        }
    }

    /**
    retrieve content from m_external_memory from an input which is usually the hidden layer activity at time t
    */
    std::vector<Expression> RNNEMBuilder::read_memory(const size_t& t, const Expression & x_t, const size_t layer)
    {
        vector<Expression> ret; 


        Expression i_w_tm1;
        if (t == 0) {
            if (has_initial_state) {
                i_w_tm1 = w0[layer];
            }
        }
        else {  // t > 0
            i_w_tm1 = w[t - 1][layer];
        }

        const vector<Expression>& vars = param_vars[layer];

        Expression M_t = vars[EXTMEM];

        /// do affine transformation to have a key
        Expression key_t = vars[WK] * x_t + vars[WKB];

        Expression raw_weight = transpose(M_t) * key_t;

        Expression beta_t = log(1.0 + exp(vars[WBETA] * x_t));

        Expression v_beta = concatenate(std::vector<Expression>(m_mem_size, beta_t));

        Expression raise_by_beta = cwise_multiply(v_beta , raw_weight);
        Expression i_alpha_t = softmax(raise_by_beta); /// get the weight to each column slice in n x 1

        Expression i_w_t;
        /// interpolation
        if (has_initial_state || t > 0)
        {
            Expression g_t = logistic(vars[WG] * x_t);
            Expression g_v = concatenate(std::vector<Expression>(m_mem_size, g_t));
            Expression f_v = concatenate(std::vector<Expression>(m_mem_size, 1.0 - g_t));
            Expression w_f_v = cwise_multiply(f_v, i_w_tm1);
            i_w_t = w_f_v + cwise_multiply(g_v, i_alpha_t);
        }
        else
        {
            i_w_t = i_alpha_t;
        }

        ret.push_back(i_w_t); /// update memory weight

        Expression retrieved_content = M_t * i_w_t;

        ret.push_back(retrieved_content);

        return ret;
    }

    Expression RNNEMBuilder::add_input_impl(const Expression& x) {
        const unsigned t = h.size();

        vector<Expression> s_tm1 = ram.final_s(); /// the ram's last state information

        w.push_back(vector<Expression>(layers));
        vector<Expression>& wt = w.back();
        Expression in = x;

        size_t i = 0;
        const vector<Expression>& vars = param_vars[i];
        Expression i_w_tm1;
        bool has_prev_state = (t > 0 || has_initial_state);
        if (t == 0) {
            if (has_initial_state) {
                // intial value for h and c at timestep 0 in layer i
                // defaults to zero matrix input if not set in add_parameter_edges
                i_w_tm1 = w0[i];
            }
        }
        else {  // t > 0
            i_w_tm1 = w[t - 1][i];
        }

        vector<Expression> mem_ret = read_memory(t, in, i);
        Expression mem_wgt = mem_ret[0];
        Expression mem_c_t = mem_ret[1];

        vector<Expression> new_in; 
        new_in.push_back(in);
        new_in.push_back(mem_c_t);
        Expression x_and_past_content = concatenate(new_in);

        Expression ram_output = ram.add_input_impl(x_and_past_content); 
        wt[i] = mem_wgt;

        return ram_output;
    } // namespace cnn


#undef WTF
#undef KTHXBYE
#undef LOLCAT
}
