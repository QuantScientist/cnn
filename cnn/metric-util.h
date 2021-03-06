#pragma once

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <math.h>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

using namespace cnn;
using namespace std;
using namespace boost::program_options;

typedef vector<cnn::real> LossStats;

class BleuMetric
{
private:
    LossStats m_allStats;

    int NgramOrder;
    int m_refIndex;
    int m_hypIndex;
    int m_matchIndex;

public:

    BleuMetric()
    {
    }

    ~BleuMetric()
    {}
    
    void AccumulateScore(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        LossStats stats = GetStats(refTokens, hypTokens);

        size_t k = 0;
        for (auto &p : stats){
            m_allStats[k++] += p;
        }
    }

    string GetScore()
    {
        cnn::real precision = Precision(m_allStats);
        cnn::real bp = BrevityPenalty(m_allStats);

        cnn::real score = precision*bp;
        return boost::lexical_cast<string>(score);
    }

    cnn::real GetSentenceScore(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        LossStats stats = GetStats(refTokens, hypTokens);
        cnn::real precision = Precision(stats);
        cnn::real bp = BrevityPenalty(stats);

        cnn::real score = precision*bp;
        return score;
    }

    LossStats GetStats(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        vector<string> lcRefTokens, lcHypTokens;
        lcRefTokens = refTokens;
        lcHypTokens = hypTokens;

        LossStats stats;
        stats.resize(1 + 2 * NgramOrder, 0.0);
        stats[m_refIndex] = ((cnn::real)lcRefTokens.size());
        for (int j = 0; j < NgramOrder; j++)
        {
            map<string, int> refCounts = GetNgramCounts(lcRefTokens, j);
            map<string, int> hypCounts = GetNgramCounts(lcHypTokens, j);

            int overlap = 0;
            for (map<string, int>::iterator e = hypCounts.begin(); e != hypCounts.end(); e++)
            {
                string ngram = e->first;
                int hypCount = e->second;
                int refCount = refCounts.count(ngram);
                overlap += min<int>(hypCount, refCount);
            }
            stats[m_hypIndex + j] = ((cnn::real)max<int>(0, lcHypTokens.size() - j));
            stats[m_matchIndex + j] = ((cnn::real)overlap);
        }
        return stats;
    }

    void Initialize(const variables_map & vm)
    {
        NgramOrder = vm["ngram_order"].as<int>(); ///default  4;
        m_allStats.resize(1 + 2 * NgramOrder, 0.0);

        m_refIndex = 0;
        m_hypIndex = m_refIndex + 1;
        m_matchIndex = m_hypIndex + NgramOrder;
    }

    void Initialize(int ngramorder = 4)
    {
        NgramOrder = ngramorder; 
        m_allStats.resize(1 + 2 * NgramOrder, 0.0);

        m_refIndex = 0;
        m_hypIndex = m_refIndex + 1;
        m_matchIndex = m_hypIndex + NgramOrder;
    }

    cnn::real BrevityPenalty(LossStats stats)
    {
        cnn::real refLen = stats[m_refIndex];
        cnn::real hypLen = stats[m_hypIndex];
        if (hypLen >= refLen)
        {
            return 1.0;
        }
        cnn::real bp = exp(1.0 - refLen / hypLen);
        return bp;
    }

private:
    std::map<string, int> GetNgramCounts(const vector<string> & tokens, int order)
    {
        map<string, int> counts;

        int n = order;

        if ((int)tokens.size() < n + 1)
            return counts; 
        
        for (int i = 0; i < (int)tokens.size() - n; i++)
        {
            string sb;
            for (int j = 0; j <= n; j++)
            {
                if (j > 0)
                {
                    sb += " ";
                }
                sb += tokens[i + j];
            }
            string ngram = sb;
            if (counts.find(ngram) == counts.end())
            {
                counts[ngram] = 1;
            }
            else
            {
                counts[ngram]++;
            }
        }

        return counts;
    }

    cnn::real Precision(LossStats stats)
    {
        cnn::real prec = 1.0;
        for (int i = 0; i < NgramOrder; i++)
        {
            cnn::real x = stats[m_matchIndex + i] / (stats[m_hypIndex + i] + 0.001);
            prec *= pow(x, 1.0 / (cnn::real)NgramOrder);
        }
        return prec;
    }

};

/// compute averaged IDF values
class IDFMetric
{
private:
    /// idf of references
    double m_refidf;

    /// idf of hypothesis
    double m_hypidf; 

    /// number of comparisons
    unsigned long m_number_comparison;

    vector<cnn::real> mv_idfs; 

public:

    IDFMetric(const vector<cnn::real>& idf_values)
    {
        mv_idfs = idf_values;
        m_number_comparison = 0; 
        m_refidf = 0;
        m_hypidf = 0;
    }

    ~IDFMetric()
    {}

    void AccumulateScore(const vector<int> & refTokens, const vector<int> & hypTokens)
    {
        pair<cnn::real, cnn::real> stats = GetStats(refTokens, hypTokens);

        m_refidf += stats.first;
        m_hypidf += stats.second; 
        m_number_comparison++;
    }

    pair<cnn::real, cnn::real> GetScore()
    {
        cnn::real refidf = 0;
        cnn::real hypidf = refidf; 
        if (m_number_comparison > 0)
        {
            refidf = m_refidf / m_number_comparison;
            hypidf = m_hypidf / m_number_comparison;
        }
        return make_pair(refidf, hypidf);
    }

    pair<cnn::real , cnn::real> GetSentenceScore(const vector<int> & refTokens, const vector<int> & hypTokens)
    {
        pair<cnn::real, cnn::real> stats = GetStats(refTokens, hypTokens);
        
        return stats;
    }

    /// compute the average of idf for reference and hypothesis
    pair<cnn::real, cnn::real> GetStats(const vector<int> & refTokens, const vector<int> & hypTokens)
    {
        cnn::real refidf = 0, hypidf = 0; 
        if (mv_idfs.size() > 0)
        {
            if (refTokens.size() > 0)
            {
                for (const auto & p : refTokens)
                    refidf += mv_idfs[p];
                refidf /= refTokens.size();
            }

            if (hypTokens.size() > 0)
            {
                for (const auto & p : hypTokens)
                    hypidf += mv_idfs[p];
                hypidf /= hypTokens.size();
            }
        }

        return make_pair(refidf, hypidf);
    }
};


/// accumulate tf-idf vectors for each input string
class TFIDFMetric
{
private:
    vector<vector<cnn::real>> m_tfidf_value;

    vector<cnn::real> mv_idfs; 
    int dim_size; 

public:

    TFIDFMetric(const vector<cnn::real>& idf_values, int dsize)
    {
        mv_idfs = idf_values;
        dim_size = dsize;
    }

    ~TFIDFMetric()
    {}

    void AccumulateScore(const vector<int> & hypTokens)
    {
        m_tfidf_value.push_back(GetStats(hypTokens));
    }

    /// compute tf-idf
    vector<cnn::real> GetStats(const vector<int> & hypTokens)
    {
        cnn::real hypidf = 0; 
    	unordered_map<int, cnn::real> tf_hyp;

        for (auto &p : hypTokens)
        {
            if (tf_hyp.find(p) == tf_hyp.end())
            {
                tf_hyp[p] = 1;
            }
            else{
                tf_hyp[p] = tf_hyp[p] + 1;
            }
        }
        
        vector<cnn::real> v_tfidf(dim_size, 0.0);
        cnn::real sum_denom = 0.0;
        for (unordered_map<int, cnn::real>::iterator ptr = tf_hyp.begin();
            ptr != tf_hyp.end();
            ptr++)
        {
            v_tfidf[ptr->first] = (1+ log( ptr->second)) * mv_idfs[ptr->first]; /// get tf-idf
            sum_denom += v_tfidf[ptr->first];
		}
        
        for (unordered_map<int, cnn::real>::iterator ptr = tf_hyp.begin();
            ptr != tf_hyp.end();
            ptr++)
        {
            v_tfidf[ptr->first] /= sum_denom; /// get tf-idf
		}


        return v_tfidf;
    }
};

namespace cnn {
    namespace metric {
        int levenshtein_distance(const std::vector<std::string> &s1, const std::vector<std::string> &s2);
        cnn::real cosine_similarity(const std::vector<cnn::real> &s1, const std::vector<cnn::real> &s2);
    }
}

/// compute averaged IDF values
class EditDistanceMetric
{
private:
    /// sum of edit distance
    double m_edit_distance;

    /// number of comparisons
    unsigned long m_number_comparison;

public:

    EditDistanceMetric()
    {
        m_number_comparison = 0;
        m_edit_distance = 0.0;
    }

    ~EditDistanceMetric()
    {}

    void AccumulateScore(const vector<string> & prvTokens, const vector<string> & hypTokens)
    {
        cnn::real stats = GetStats(prvTokens, hypTokens);

        m_edit_distance += stats;
        m_number_comparison++;
    }

    cnn::real GetScore()
    {
        cnn::real edist = 0;
        if (m_number_comparison > 0)
        {
            edist = m_edit_distance / m_number_comparison;
        }
        return edist;
    }

    cnn::real GetSentenceScore(const vector<string>& prev_response, const vector<string> & hypTokens)
    {
        cnn::real stats = GetStats(prev_response, hypTokens);

        return stats;
    }

    /// compute the edit distance between two strings
    cnn::real GetStats(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        cnn::real edtdistance = metric::levenshtein_distance(refTokens, hypTokens); 
        
        return edtdistance;
    }
};

