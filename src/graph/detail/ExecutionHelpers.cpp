/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

#include "arm_compute/runtime/CPUUtils.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

namespace arm_compute
{
namespace graph
{
namespace detail
{
void validate_all_nodes(Graph &g)
{
    auto &nodes = g.nodes();

    // Create tasks
    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target                    assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            Status                    status          = backend.validate_node(*node);
            ARM_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
        }
    }
}

unsigned int _split = 1; //split from layer X

void configure_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && tensor->handle() == nullptr)
        {
            Target                         target  = tensor->desc().target;
            backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
            std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }


    // split is read from file       
    std::ifstream iffactor;	    
    iffactor.open("/root/.hikey960/split", std::ios::in);	        
    if(iffactor.is_open()){			            
	    std::string line;
	    while(bool(getline(iffactor,line))) {
		    if(line.empty()) continue;
		    _split = std::stod(line,nullptr);
	    }
    }
    iffactor.close();
    std::cout << "Setting split at layer: " << _split << std::endl;
}

void allocate_all_input_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        Tensor *tensor = node.input(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_all_output_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *tensor = node.output(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_const_tensors(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
            }
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr && tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
        {
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order)
{
    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    // Create tasks
    for(auto &node_id : node_order)
    {
        auto node = g.node(node_id);
        if(node != nullptr)
        {
            Target                     assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            std::unique_ptr<IFunction> func            = backend.configure_node(*node, ctx);
            if(func != nullptr)
            {
                ExecutionTask task;
                task.task = std::move(func);
                task.node = node;
                workload.tasks.push_back(std::move(task));
            }
        }
    }

    // Add inputs and outputs
    for(auto &node : g.nodes())
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            continue;
        }
    }

    return workload;
}

void release_unused_tensors(Graph &g)
{
    for(auto &tensor : g.tensors())
    {
        if(tensor != nullptr && tensor->handle() != nullptr)
        {
            tensor->handle()->release_if_unused();
        }
    }
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const)
        {
            call_tensor_accessor(node->output(0));
        }
    }
}

bool call_all_input_node_accessors(ExecutionWorkload &workload)
{
    return !std::any_of(std::begin(workload.inputs), std::end(workload.inputs), [](Tensor * input_tensor)
    {
        return (input_tensor == nullptr) || !input_tensor->call_accessor();
    });
}

void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    for(auto &task : workload.tasks)
    {
        task.prepare();
        release_unused_tensors(*workload.graph);
    }
}


std::mutex allc_mutex_0;
std::mutex allc_mutex_4;
std::mutex common_mutex;


void call_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);

 //   static double total=0, active=0, batch=0, conv=0, dcon =0, depth_conv=0, elt=0, flat=0, fc=0, norm=0, pool=0, reshape=0, scale=0, smax=0, split=0;

    // Acquire memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }

        cpu_set_t cpuset0;	    
	cpu_set_t cpuset4;

	CPU_ZERO(&cpuset0);
	CPU_ZERO(&cpuset4);

	CPU_SET(0, &cpuset0);
	CPU_SET(4, &cpuset4);

	pthread_t current_thread;

    // Execute tasks
    for(auto &task : workload.tasks)
    {



	if(workload.graph->id().get() == 2) //for CL implementation
	{
		task();

	}else {

		if (&task == &workload.tasks.front()){
		    allc_mutex_4.lock();
		    current_thread = pthread_self();
		    int rc= pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset4);
		    if (rc !=0) std::cout << "Error in setting affinity for thread " << sched_getcpu() << std::endl;
	    }
	    if(task.node->id() == _split) {
		    allc_mutex_4.unlock();
		    allc_mutex_0.lock();
		    current_thread = pthread_self();
		    int rc= pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset0);   
		    if (rc !=0) std::cout << "Error in setting affinity for thread " << sched_getcpu() << std::endl;
	    }


	    
	    
//	auto tbegin = std::chrono::high_resolution_clock::now();
        task();
//	auto tend = std::chrono::high_resolution_clock::now();	
//	double cost = std::chrono::duration_cast<std::chrono::duration<double>> (tend - tbegin).count();

	 if (&task == &workload.tasks.back()){	
		 allc_mutex_0.unlock();
	 }

	} //else end

	std::cout << "Node info: ID: " << task.node->id() << ", type: " << (int) task.node->type() << "Graph: "<< workload.graph->id().get() << std::endl;
/*	std::cout << "Node info: ID: " << task.node->id() << ", type: " << (int) task.node->type() << ", cost: "  << cost << std::endl;
	total += cost;
	if((int)task.node->type() == 0) active += cost;
	if((int)task.node->type() == 1) batch += cost;
	if((int)task.node->type() == 4) conv += cost;
	if((int)task.node->type() == 5) dcon += cost;
	if((int)task.node->type() == 6) depth_conv += cost;
	if((int)task.node->type() == 7) elt += cost;
	if((int)task.node->type() == 8) flat += cost;
	if((int)task.node->type() == 9) fc += cost;
	if((int)task.node->type() == 10) norm += cost;
	if((int)task.node->type() == 12) pool += cost;
	if((int)task.node->type() == 13) reshape += cost;
	if((int)task.node->type() == 14) scale += cost;
	if((int)task.node->type() == 15) smax += cost;
	if((int)task.node->type() == 16) split += cost;
*/

    } //for end

 /*   std::cout << "Summary: " << std::endl
	    << "Total time: " << total << std::endl
	    << "ActivationLayer: " << active << std::endl
	    << "BatchNormalizationLayer: " << batch << std::endl
	    << "ConvolutionLayer: " << conv << std::endl
	    << "DeconvolutionLayer: " << dcon << std::endl
	    << "DepthwiseConvolutionLayer: " << depth_conv << std::endl
	    << "EltwiseLayer: " << elt << std::endl
	    << "FlattenLayer: " << flat << std::endl
	    << "FullyConnectedLayer: " << fc << std::endl
	    << "NormalizationLayer: " << norm << std::endl
	    << "PoolingLayer: " << pool << std::endl
	    << "ReshapeLayer: " << reshape << std::endl
	    << "ResizeLayer: " << scale << std::endl
	    << "SoftmaxLayer: " << smax << std::endl
	    << "SplitLayer: " << split << std::endl;
*/

    // Release memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

bool call_all_output_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.outputs), std::end(workload.outputs), [&](Tensor * output_tensor)
    {
        is_valid = is_valid && (output_tensor != nullptr) && output_tensor->call_accessor();
    });

    return is_valid;
}
} // namespace detail
} // namespace graph
} // namespace arm_compute
