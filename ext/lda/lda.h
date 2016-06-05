#ifndef	_ldaModel_H
#define	_ldaModel_H

/**
    the implementation of LDA refers to the simple LDA in dmlc/experimental-lda
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cnn/data-util.h>
#include <cnn/dict.h>
#include <cnn/math.h>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

class ldaModel {
public:

	/****** constructor/destructor ******/
	ldaModel(const Corpus& train, const Corpus& test);
	~ldaModel();

	/****** interacting functions ******/
	ldaModel* init(variables_map vm);// initialize the ldaModel randomly
	int train();					// train LDA using prescribed algorithm on training data
	int test(Dict& sd);						// test LDA according to specified method
    int test(Dict& sd, const SentencePair& obs);

    /****** Initialisation aux ******/
    int read_data(const Corpus & training, const Dict& sd, const Corpus& test);				// Read training (and testing) data

    int save_ldaModel_topWords(std::string filename, Dict& sd) const;// ldaModel_name.twords: Top words in each top
    int load_ldaModel(int iter);
    int load_ldaModel(const string & filename);

    int topic_of(int m);

    friend class boost::serialization::access;
    template<class Archive> void save(Archive& ar, const unsigned int version) const {
        ar & alpha;
        ar & beta;
        ar & K; 
        ar & M; 
        ar & V;
        ar & n_iters;

        for (int v = 0; v < V; v++)
            ar & n_wk[v];
        ar & n_k;
    }
    template<class Archive> void load(Archive& ar, const unsigned int version) {
        ar & alpha;
        ar & beta;
        ar & K;
        ar & M;
        ar & V;
        ar & n_iters;

        n_wk.resize(V);
        for (int v = 0; v < V; v++)
            ar & n_wk[v];
        ar & n_k;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
	/****** Enums for testing type ldaModel status  ******/
	enum {							// testing types:
		INVALID,					// ldaModel not initialised
		NO_TEST,					// do not report any likelihood
		SELF_TEST,					// report training likelihood
		SEPARATE_TEST				// report likelihood on a held-out testing set
	} testing_type;

	/****** DATA ******/
	Sentences trngdata;				// training dataset
	Sentences testdata;				// test dataset
    Sentences trngresponses;         // responses
    Sentences testresponses;         // responses

    /****** ldaModel Parameters ******/
	int M;							// Number of documents
	int V; 							// Number of words in dictionary
	int K; 							// Number of topics

	/****** ldaModel Hyper-Parameters ******/
	double alpha;					// per document Topic proportions dirichlet prior
	double beta, Vbeta;				// Dirichlet language ldaModel

	/****** ldaModel variables ******/
	int ** z;						// topic assignment for each word
	std::vector<vector<int>> n_wk;					// number of times word w assigned to topic k
	std::vector< std::vector< std::pair<int, int> > > n_mks; //sparse representation of n_mk: number of words assigned to topic k in document m
	std::vector<int> n_k;						// number of words assigned to topic k = sum_w n_wk = sum_m n_mk
		
	/****** Temporary variables ******/
	double * p;
	int *nd_m;
	int *rev_mapper;
      
	/****** Training aux ******/
	int n_iters;	 				// Number of Gibbs sampling iterations
	int n_save;			 			// Number of iters in between saving
	int n_topWords; 				// Number of top words to be printed per topic
	int init_train();					// init for training
	virtual int specific_init() { return 0; }	// if sampling algo need some specific inits
    int sampling(int m);// sampling doc m outsourced to children
    
	/****** Testing aux ******/
	int test_n_iters;
	int test_M;
	int ** test_z;
	int ** test_n_wk;
	int ** test_n_mk;
	int * test_n_k;
	int init_test();				// init for testing
	int vanilla_sampling(int m);	// vanila sampling doc m for testing
    int vanilla_sampling(const Sentence& obs);

	/****** Functions to update sufficient statistics ******/
	inline int add_to_topic(int w, int m, int topic, int old_topic)
	{
        n_wk[w][topic] += 1; 
		if (topic != old_topic && nd_m[topic] == 0)
		{
			rev_mapper[topic] = n_mks[m].size();
			n_mks[m].push_back(std::pair<int, int>(topic, 1));
		}
		else
			n_mks[m][rev_mapper[topic]].second += 1;
		nd_m[topic] += 1;
		if (nd_m[old_topic] == 0)
		{
			n_mks[m][rev_mapper[old_topic]].first = n_mks[m].back().first;
			n_mks[m][rev_mapper[old_topic]].second = n_mks[m].back().second;
			rev_mapper[n_mks[m].back().first] = rev_mapper[old_topic];
			n_mks[m].pop_back();
			rev_mapper[old_topic] = -1;
		}
        n_k[topic] += 1; 

		return 0;
	}
	inline int remove_from_topic(int word, int doc, int topic)
	{
        n_wk[word][topic] -= 1; 
		nd_m[topic] -= 1;
		n_mks[doc][rev_mapper[topic]].second -= 1;
        n_k[topic] -= 1; 

		return 0;
	}

	/****** Performance computations ******/
	std::vector<double> time_ellapsed; // time ellapsed after each iteration
	std::vector<double> likelihood; // likelihood after each iteration
	double newllhw() const;			// per word log-likelihood for new (unseen) data based on the estimated LDA ldaModel
	double llhw() const;			// per word log-likelihood for training data based on the estimated LDA ldaModel

	/****** File and Folder Paths ******/
	std::string ddir;				// data directory
	std::string mdir;				// ldaModel directory
	const Corpus& tfile;				// test data corpus 
    const Corpus& dfile;              // train data corpus

	/****** save LDA ldaModel to files ******/
	int save_ldaModel(int iter) const;						// save ldaModel: call each of the following:		
	int save_ldaModel_time(std::string filename) const;	// ldaModel_name.time: time at which statistics calculated
	int save_ldaModel_llh(std::string filename) const;		// ldaModel_name.llh: Per word likelihood on held out documents
	int save_ldaModel_phi(std::string filename) const;		// ldaModel_name.phi: topic-word distributions

	/****** Some functions for debugging ******/
	int sanity() const;
};

#endif