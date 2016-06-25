/**
The acoustic model header 
*/

#pragma once

#include <cnn/data-util.h>
#include <ext/acoustics/acoustic_model.h>

template <class DBuilder>
class ASR{
public:
    DBuilder s2tmodel;  /// source to target 

    int swords;
    int twords;
    int nbr_turns;
    int hidden_replicate;

    vector<unsigned> hidden_dim;

    Expression s2txent;

public:
    explicit ASR(cnn::Model& model,
        const vector<unsigned int>& layers,
        unsigned vocab_size_tgt,
        const vector<unsigned>& hidden_dim,
        map<string, cnn::real>& additional_argument,
        cnn::real iscale = 1.0)
        : s2tmodel(model, vocab_size_tgt, layers, hidden_dim, 
        additional_argument,
        (int) additional_argument["replicatehidden"],
        (int) additional_argument["decoder_use_additional_input"],
        iscale), hidden_dim(hidden_dim)
    {
        swords = 0;
        twords = 0;
        nbr_turns = 0;

        hidden_replicate = (int) additional_argument["replicatehidden"];

        if (verbose)
            cout << "start ASR class" << endl;
    }

    // return Expression of total loss
    // only has one pair of sentence so far
    vector<Expression> build_graph(const PRealVectorObsAndItsLabelsTurn& cur_sentence,
        ComputationGraph& cg) 
    {
        vector<Expression> object;

        if (verbose)
            cout << "start MultiSourceDialogue:build_graph(const Dialogue & cur_sentence, ComputationGraph& cg)" << endl;

        twords = 0;
        swords = 0;
        nbr_turns = 1;
        vector<vector<cnn::real>> insent;
        vector<Sentence> osent, prv_response;
        for (auto p : cur_sentence)
        {
            insent.push_back(p.first);
            osent.push_back(p.second);

            twords += p.second.size();
            swords += p.first.size();
        }

        object = s2tmodel.build_graph(insent, osent, cg);
        if (verbose)
            display_value(object.back(), cg, "object");

        s2txent = sum(object);
        if (verbose)
            display_value(s2txent, cg, "s2txent");

//        s2tmodel.serialise_context(cg);

        if (twords != s2tmodel.tgt_words || swords != s2tmodel.src_feature_number)
        {
            string msg = "from corpus: twords = " + boost::lexical_cast<string>(twords) + " obs size = " + boost::lexical_cast<string>(swords);
            string omsg = " from model builder: lables = " + boost::lexical_cast<string>(s2tmodel.tgt_words) + " obs size = " + boost::lexical_cast<string>(s2tmodel.src_feature_number); 
            string nmsg = " possible reason : embedding dimension in the model and the obs feature dimension mismatch";

            throw(msg + omsg + nmsg);
        }

        return object;
    }

    void reset()
    {
        s2tmodel.reset();
    }

    string respond(const vector<cnn::real>& obs, Dict & td)
    {
        string shuman;
        string response;
        unsigned lines = 0;

        vector<int> decode_output;

        ComputationGraph cg;
        s2tmodel.reset();
        decode_output = s2tmodel.decode(obs, cg, td);

        if (verbose)
        {
            cout << "Agent: ";
            response = "";
        }

        for (auto pp : decode_output)
        {
            if (verbose)
            {
                cout << td.Convert(pp) << " ";
            }

            if (pp != kSRC_EOS && pp != kSRC_SOS)
                response = response + td.Convert(pp) + " ";

            if (verbose)
            {
                cout << " ";
            }
        }

        if (verbose)
            cout << endl;
        return response;
    }

};

template <class DBuilder>
class AcousticClassification{
public:
    DBuilder s2tmodel;  /// source to target 

    int swords;
    int twords;
    int nbr_turns;
    int hidden_replicate;

    vector<unsigned> hidden_dim;

    Expression s2txent;

public:
    explicit AcousticClassification(cnn::Model& model,
        const vector<unsigned int>& layers,
        unsigned vocab_size_tgt,
        const vector<unsigned>& hidden_dim,
        map<string, cnn::real>& additional_argument,
        cnn::real iscale = 1.0)
        : s2tmodel(model, vocab_size_tgt, layers, hidden_dim,
        additional_argument,
        (int)additional_argument["replicatehidden"],
        (int)additional_argument["decoder_use_additional_input"],
        iscale), hidden_dim(hidden_dim)
    {
        swords = 0;
        twords = 0;
        nbr_turns = 0;

        hidden_replicate = (int)additional_argument["replicatehidden"];

        if (verbose)
            cout << "start ASR class" << endl;
    }

    // return Expression of total loss
    // only has one pair of sentence so far
    vector<Expression> build_graph(const PRealVectorObsAndItsLabelsTurn& cur_sentence,
        ComputationGraph& cg)
    {
        vector<Expression> object;

        if (verbose)
            cout << "start MultiSourceDialogue:build_graph(const Dialogue & cur_sentence, ComputationGraph& cg)" << endl;

        twords = 0;
        swords = 0;
        nbr_turns = 1;
        vector<vector<cnn::real>> insent;
        vector<Sentence> osent, prv_response;
        for (auto p : cur_sentence)
        {
            insent.push_back(p.first);
            osent.push_back(p.second);

            twords += p.second.size();
            swords += p.first.size();
        }

        object = s2tmodel.build_graph(insent, osent, cg);
        if (verbose)
            display_value(object.back(), cg, "object");

        s2txent = sum(object);
        if (verbose)
            display_value(s2txent, cg, "s2txent");

        if (twords != s2tmodel.tgt_words || swords != s2tmodel.src_feature_number)
        {
            string msg = "from corpus: twords = " + boost::lexical_cast<string>(twords)+" obs size = " + boost::lexical_cast<string>(swords);
            string omsg = " from model builder: lables = " + boost::lexical_cast<string>(s2tmodel.tgt_words) + " obs size = " + boost::lexical_cast<string>(s2tmodel.src_feature_number);
            string nmsg = " possible reason : embedding dimension in the model and the obs feature dimension mismatch";

            throw(msg + omsg + nmsg);
        }

        return object;
    }

    void reset()
    {
        s2tmodel.reset();
    }

    string respond(const vector<cnn::real>& obs, Dict & td)
    {
        string shuman;
        string response;
        unsigned lines = 0;

        vector<int> decode_output;

        ComputationGraph cg;
        s2tmodel.reset();
        decode_output = s2tmodel.decode(obs, cg, td);

        if (verbose)
        {
            cout << "Agent: ";
            response = "";
        }

        for (auto pp : decode_output)
        {
            if (verbose)
            {
                cout << td.Convert(pp) << " ";
            }

            if (pp != kSRC_EOS && pp != kSRC_SOS)
                response = response + td.Convert(pp) + " ";

            if (verbose)
            {
                cout << " ";
            }
        }

        if (verbose)
            cout << endl;
        return response;
    }

};
