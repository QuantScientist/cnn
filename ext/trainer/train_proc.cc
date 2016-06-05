#include "ext/trainer/train_proc.h"

using namespace std;

cnn::real smoothed_ppl(cnn::real curPPL, vector<cnn::real>& ppl_hist)
{
    if (ppl_hist.size() == 0)
        ppl_hist.resize(3, curPPL);
    ppl_hist.push_back(curPPL);
    if (ppl_hist.size() > 3)
        ppl_hist.erase(ppl_hist.begin());

    cnn::real finPPL = 0;
    size_t k = 0;
    for (auto p : ppl_hist)
    {
        finPPL += p;
        k++;
    }
    return finPPL/k;
}

void reset_smoothed_ppl(vector<cnn::real>& ppl_hist)
{
     ppl_hist.clear();
}
