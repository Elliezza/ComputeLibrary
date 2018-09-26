/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"


#include <sched.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph0(0, "SqueezeNetV1")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

	graph0 << common_params.target;
	graph1 << common_params.target;
	graph2 << Target::CL;

	initialize_graph(graph0);
	initialize_graph(graph1);
	initialize_graph(graph2);

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_file  = common_params.tuner_file;

	graph0.finalize(common_params.target, config);
	graph1.finalize(common_params.target, config);
	graph2.finalize(Target::CL, config);


        return true;
    }

    void do_run() override
    {
	    std::cout << "Starting of running the kernel" << std::endl; 

	    int num_cores = 3;
	    int k = 4;
	    std::vector<std::thread> workers(num_cores);

	    auto tbegin = std::chrono::high_resolution_clock::now();
	    for(int i = 0; i < num_cores; ++i){
		    workers[i] = std::thread([&, i]{
				    {
				    std::cout << "Creating new threads: " << i << " on CPU:" << sched_getcpu() << std::endl;
				    if (i == 0) {
				    for (int j = 0; j < 8; j++) graph0.run();
				    } else if (i == 1){
				    for (int j = 0; j < 8; j++) graph1.run();
				    } else if (i == 2){
				    for (int j = 0; j < 40; j++) graph2.run();
				    }
				    }});
		    cpu_set_t cpuset;
		    CPU_ZERO(&cpuset);
		    if (i==2) {
			    CPU_SET(i, &cpuset);
		    } 
		    else{
			    CPU_SET((k+i), &cpuset);}
		    int rc= pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		    if (rc !=0) std::cout << "Error in setting affinity for thread " << i << std::endl;
	    }
	    for(auto&t: workers) t.join();

	    auto tend = std::chrono::high_resolution_clock::now();
	    double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
	    double cost = cost0/56;

	    std::cout << "COST:" << cost << std::endl;
    }


private:
    CommandLineParser   cmd_parser;
    CommonGraphOptions  common_opts;
    CommonGraphParams   common_params;
    Stream              graph0;
    Stream		graph1{1, "SqueezeNetV1"};
    Stream              graph2{2, "SqueezeNetV1"};
	    
	   
    void initialize_graph(Stream &graph)
    {
	    std::string data_path = common_params.data_path;

	    const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
	    std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

	    const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
	    TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

	    const DataLayout weights_layout = DataLayout::NCHW;

	  //  graph << common_params.target
	
	    graph   << common_params.fast_math_hint
		    << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
		    << ConvolutionLayer(
				    7U, 7U, 96U,
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy", weights_layout),
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
				    PadStrideInfo(2, 2, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		    << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		    << ConvolutionLayer(
				    1U, 1U, 16U,
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy", weights_layout),
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
				    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire2", weights_layout, 64U, 64U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 16U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire3", weights_layout, 64U, 64U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 32U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire4", weights_layout, 128U, 128U, graph);
	    graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		    << ConvolutionLayer(
				    1U, 1U, 32U,
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy", weights_layout),
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
				    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire5", weights_layout, 128U, 128U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 48U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire6", weights_layout, 192U, 192U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 48U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire7", weights_layout, 192U, 192U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 64U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire8", weights_layout, 256U, 256U, graph);
	    graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		    << ConvolutionLayer(
				    1U, 1U, 64U,
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy", weights_layout),
				    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
				    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	    graph << get_expand_fire_node(data_path, "fire9", weights_layout, 256U, 256U, graph);
	    graph << ConvolutionLayer(
			    1U, 1U, 1000U,
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy", weights_layout),
			    get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
			    PadStrideInfo(1, 1, 0, 0))
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		    << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
		    << FlattenLayer()
		    << SoftmaxLayer()
		    << OutputLayer(get_output_accessor(common_params, 5));

    
    }

    BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int expand1_filt, unsigned int expand3_filt, Stream &graph )
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }
};

/** Main program for Squeezenet v1.0
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
}
