#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <random>

using namespace std;
typedef vector<vector<long double>> matrix;

enum ActivationType{
    Sigmoid,
    Liner,
    SoftMax,
    Relu
};

class Activation
{
    matrix Sigmoid(matrix &x)
    {
        matrix res;

        for(auto batch : x)
        {
            vector<long double> t;

            for(auto i : batch)
            {
                t.push_back(1.0 / (1.0 + exp(-i)));
            }

            res.push_back(t);
        }

        return res;
    }

    matrix Relu(matrix &x)
    {
        matrix res;

        for(auto batch : x)
        {
            vector<long double> t;

            for(auto i : batch)
            {
                t.push_back((i >= 0) * i);
            }

            res.push_back(t);
        }

        return res;
    }

    matrix Softmax(matrix &x)
    {
        matrix res;

        for(auto batch : x)
        {
            vector<long double> t;
            long double c = *max_element(batch.begin(), batch.end());
            long double sum = 0;

            for (long double i : batch)
            {
                sum += exp(i - c);
            }
            for(auto i : batch)
            {
                t.push_back(exp(i - c) / sum);
            }

            res.push_back(t);
        }

        return res;
    }

    ActivationType m_name;

public:
    Activation();
    Activation(ActivationType name) : m_name(name) {}
    
    matrix forward(matrix &x)
    {
        if (m_name == ActivationType::Sigmoid) return Sigmoid(x);
        else if (m_name == ActivationType::Relu) return Relu(x);
        else if (m_name == ActivationType::SoftMax) return Softmax(x);
        else return x;
    }
};

class Dense
{
    Activation m_activation;
    vector<long double> bias;
    matrix neuron;

public:
    Dense(int input_unit, int unit, ActivationType activation):

        bias(unit),
        neuron(unit, std::vector<long double>(input_unit)),
        m_activation(activation)
        {

            double sigma = 0.05;
            if(activation == ActivationType::Relu) sigma = sqrt(2.0 / (double)input_unit);
            else if(activation == ActivationType::Sigmoid || activation == ActivationType::SoftMax) sigma = sqrt(1.0 / (double)input_unit);
            else sigma = 0.05;

            // 重みとバイアスを初期化
            random_device seed;
            mt19937 engine(seed());
            normal_distribution<> generator(0.0, sigma);

            for(int i = 0; i < unit; ++i){
                bias[i] = generator(engine);
                for(int j = 0; j < input_unit; ++j){
                    neuron[i][j] = generator(engine);
                }
            }
        }

    matrix forward(matrix &data)
    {
        matrix ans;

        for (int index = 0; auto &i : data){

            std::vector<long double> res;
            for (int j = 0; j < neuron.size(); ++j){
                long double t = 0;
                // 入力 * 重み
                for (int k = 0; k < neuron[j].size(); ++k){
                    t += i[k] * neuron[j][k];
                }

                // バイアスの適用
                t -= bias[j];

                res.push_back(t);
            }

            ans.push_back(res);

            ++index;
        }

        ans = m_activation.forward(ans);
    }
};

class Model
{
    int m_input_size, m_output_size;
    vector<Dense> model;
public:
    Model(int input_size):m_input_size(input_size),m_output_size(input_size){}
    void AddDenseLayer(int unit, ActivationType activation){
        Dense dense(m_output_size, unit, activation);
        model.push_back(dense);
        m_output_size = unit;
    }
    matrix predict(matrix &data)
    {
        matrix res = data;
        for(auto &layer :model)
        {
            res = layer.forward(res);
        }

        return res;
    }
};


int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}