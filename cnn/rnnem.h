#ifndef CNN_RNNEM_H_
#define CNN_RNNEM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/gru.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

class Model;

/// to-do : expose this to be a variable so that it can be changed 
/// this number needs to be used in attentional.h so has to be defined here so that both RNNEM and attentional model know the 
/// memory size, which is the number of columns
/// the momory row dimension is the same as hidden layer dimension
#define RNNEM_MEM_SIZE 512

struct RNNEMBuilder : public RNNBuilder{

    friend struct GRUBuilder;
    RNNEMBuilder() = default;
  explicit RNNEMBuilder(long layers,
      long input_dim,
      long hidden_dim,
      Model* model);

  void rewind_one_step() {
      ram.rewind_one_step(); 
      w.pop_back();
  }
  Expression back() const { 
      vector<Expression> tmp_back = final_h();
      return concatenate(tmp_back); 
  }
  std::vector<Expression> final_h() const 
  {
      std::vector<Expression> ret = final_w();
      vector<Expression> frm_ram = ram.final_h(); 
      for (auto p = frm_ram.begin(); p != frm_ram.end(); p++)
          ret.push_back(*p); 

      return ret; 
  }
  std::vector<Expression> final_w() const { return (w.size() == 0 ? w0 : w.back()); }
  std::vector<Expression> final_s() const {
      return final_h(); 
  }
 private:
     std::vector<Expression> read_memory(const size_t& t, const Expression & x_t, const size_t layer);
     GRUBuilder ram; 

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(const Expression& x) override;

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h, w;
  // for external memeory
  std::vector<std::vector<Expression>> M;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> w0, h0, c0;
  long layers;
  long m_mem_size;
  long m_mem_dim;
};

//template class RNNEMBuilder<GRUBuilder>;
//template class RNNEMBuilder<LSTMBuilder>;

} // namespace cnn

#endif
