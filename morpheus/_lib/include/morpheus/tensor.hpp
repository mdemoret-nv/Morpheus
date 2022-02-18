/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cudf/types.hpp>
#include <memory>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/device_buffer.hpp>
#include <trtlab/neo/core/tensor.hpp>
#include "pyneo/node.hpp"

#include "morpheus/matx_functions.hpp"
#include "morpheus/type_utils.hpp"

namespace morpheus {

namespace neo   = trtlab::neo;
namespace py    = pybind11;
namespace pyneo = trtlab::neo::pyneo;

struct ITensor : public neo::ITensor
{
    virtual bool is_host_accessible() const   = 0;
    virtual bool is_device_accessible() const = 0;

    virtual std::shared_ptr<ITensor> tensor_like(const DType& dtype,
                                                 std::vector<neo::TensorIndex> shape,
                                                 std::vector<neo::TensorIndex> strides,
                                                 size_t offset = 0) const = 0;
};

template <typename TensorT>
struct Tensor
{
    // static neo::TensorObject create(std::shared_ptr<rmm::device_buffer> buffer,
    //                                 DType dtype,
    //                                 std::vector<neo::TensorIndex> shape,
    //                                 std::vector<neo::TensorIndex> strides,
    //                                 size_t offset = 0)
    // {
    //     auto md = nullptr;

    //     auto tensor = std::make_shared<RMMTensor>(buffer, offset, dtype, shape, strides);

    //     return neo::TensorObject(md, tensor);
    // }
};

template <typename TensorT>
struct TensorAdaptor
{
    static constexpr bool IsHostAccessible   = false;
    static constexpr bool IsDeviceAccessible = false;

    static std::shared_ptr<TensorT> reshape(const TensorT& self, const std::vector<neo::TensorIndex>& dims)
    {
        throw std::runtime_error("Not implemented");
    }

    static std::shared_ptr<TensorT> tensor_like(const TensorT& self,
                                                const DType& dtype,
                                                std::vector<neo::TensorIndex> shape,
                                                std::vector<neo::TensorIndex> strides,
                                                size_t offset = 0)
    {
        throw std::runtime_error("Not implemented");
    }
};

struct TensorView
{
    TensorView() = default;

    TensorView(std::shared_ptr<ITensor> tensor) : m_tensor(std::move(tensor)) {}

    TensorView(const TensorView& other) = default;

    TensorView(TensorView&& other) : m_tensor(std::exchange(other.m_tensor, nullptr)) {}

    ~TensorView() = default;

    void* data() const
    {
        return m_tensor->data();
    }

    neo::DataType dtype() const
    {
        return m_tensor->dtype();
    }

    std::size_t count() const
    {
        return m_tensor->count();
    }
    std::size_t bytes() const
    {
        return m_tensor->bytes();
    }

    neo::RankType rank() const
    {
        return m_tensor->rank();
    }
    std::size_t dtype_size() const
    {
        return m_tensor->dtype().item_size();
    }

    std::vector<neo::TensorIndex> get_shape() const
    {
        std::vector<neo::TensorIndex> s;

        m_tensor->get_shape(s);

        return s;
    }

    std::vector<neo::TensorIndex> get_stride() const
    {
        std::vector<neo::TensorIndex> s;

        m_tensor->get_stride(s);

        return s;
    }

    neo::TensorIndex shape(std::uint32_t idx) const
    {
        return m_tensor->shape(idx);
    }
    neo::TensorIndex stride(std::uint32_t idx) const
    {
        return m_tensor->stride(idx);
    }

    void get_shape(std::vector<neo::TensorIndex>& s) const
    {
        return m_tensor->get_shape(s);
    }
    void get_stride(std::vector<neo::TensorIndex>& s) const
    {
        return m_tensor->get_stride(s);
    }

    bool is_compact() const
    {
        return m_tensor->is_compact();
    }

    // TensorObject slice(std::vector<neo::TensorIndex> min_dims, std::vector<neo::TensorIndex> max_dims) const
    // {
    //     // Replace any -1 values
    //     std::replace_if(
    //         min_dims.begin(), min_dims.end(), [](auto x) { return x < 0; }, 0);
    //     std::transform(
    //         max_dims.begin(), max_dims.end(), this->get_shape().begin(), max_dims.begin(), [](auto d, auto s) {
    //             return d < 0 ? s : d;
    //         });

    //     return TensorAdaptor<typename TensorT>

    //         return TensorObject(m_tensor->slice(min_dims, max_dims));
    // }

    // TensorObject reshape(const std::vector<neo::TensorIndex>& dims) const
    // {
    //     return TensorObject(m_tensor->reshape(dims));
    // }

    // TensorObject deep_copy() const
    // {
    //     std::shared_ptr<neo::ITensor> copy = m_tensor->deep_copy();

    //     return TensorObject(copy);
    // }

    // std::vector<uint8_t> get_host_data() const
    // {
    //     std::vector<uint8_t> out_data;

    //     out_data.resize(this->bytes());

    //     NEO_CHECK_CUDA(cudaMemcpy(&out_data[0], this->data(), this->bytes(), cudaMemcpyDeviceToHost));

    //     return out_data;
    // }

    template <typename T, size_t N>
    T read_element(const neo::TensorIndex (&idx)[N]) const
    {
        auto stride = this->get_stride();
        auto shape  = this->get_shape();

        CHECK(std::transform_reduce(
            stride.begin(), stride.end(), std::begin(idx), 0, std::logical_and<>(), std::less<>()))
            << "Index is outsize of the bounds of the tensor. Index="
            << neo::detail::array_to_str(std::begin(idx), std::begin(idx) + N)
            << ", Size=" << neo::detail::array_to_str(shape.begin(), shape.end()) << "";

        CHECK(neo::DataType::create<T>() == this->dtype())
            << "read_element type must match array type. read_element type: '" << neo::DataType::create<T>().name()
            << "', array type: '" << this->dtype().name() << "'";

        size_t offset = std::transform_reduce(
                            stride.begin(), stride.end(), std::begin(idx), 0, std::plus<>(), std::multiplies<>()) *
                        this->dtype_size();

        T output;

        NEO_CHECK_CUDA(
            cudaMemcpy(&output, static_cast<uint8_t*>(this->data()) + offset, sizeof(T), cudaMemcpyDeviceToHost));

        return output;
    }

    template <typename T, size_t N>
    T read_element(const std::array<neo::TensorIndex, N> idx) const
    {
        auto stride = this->get_stride();
        auto shape  = this->get_shape();

        CHECK(std::transform_reduce(
            stride.begin(), stride.end(), std::begin(idx), 0, std::logical_and<>(), std::less<>()))
            << "Index is outsize of the bounds of the tensor. Index="
            << neo::detail::array_to_str(std::begin(idx), std::begin(idx) + N)
            << ", Size=" << neo::detail::array_to_str(shape.begin(), shape.end()) << "";

        CHECK(neo::DataType::create<T>() == this->dtype())
            << "read_element type must match array type. read_element type: '" << neo::DataType::create<T>().name()
            << "', array type: '" << this->dtype().name() << "'";

        size_t offset = std::transform_reduce(
                            stride.begin(), stride.end(), std::begin(idx), 0, std::plus<>(), std::multiplies<>()) *
                        this->dtype_size();

        T output;

        NEO_CHECK_CUDA(
            cudaMemcpy(&output, static_cast<uint8_t*>(this->data()) + offset, sizeof(T), cudaMemcpyDeviceToHost));

        return output;
    }

    // move assignment
    TensorView& operator=(TensorView&& other) noexcept
    {
        // Guard self assignment
        if (this == &other)
            return *this;

        m_tensor = std::exchange(other.m_tensor, nullptr);
        return *this;
    }

    // copy assignment
    TensorView& operator=(const TensorView& other)
    {
        // Guard self assignment
        if (this == &other)
            return *this;

        // Check for valid assignment
        if (this->get_shape() != other.get_shape())
        {
            throw std::runtime_error("Left and right shapes do not match");
        }

        if (this->get_stride() != other.get_stride())
        {
            throw std::runtime_error(
                "Left and right strides do not match. At this time, only uniform strides are allowed");
        }

        // Inefficient but should be sufficient
        if (this->get_numpy_typestr() != other.get_numpy_typestr())
        {
            throw std::runtime_error("Left and right types do not match");
        }

        DCHECK(this->bytes() == other.bytes()) << "Left and right bytes should be the same if all other test passed";

        // Perform the copy operation
        NEO_CHECK_CUDA(cudaMemcpy(this->data(), other.data(), this->bytes(), cudaMemcpyDeviceToDevice));

        return *this;
    }

    // std::shared_ptr<neo::MemoryDescriptor> get_memory() const
    // {
    //     return m_md;
    // }

    std::string get_numpy_typestr() const
    {
        return m_tensor->dtype().type_str();
    }

    TensorView as_type(neo::DataType dtype) const
    {
        if (dtype == m_tensor->dtype())
        {
            // Shallow copy
            return TensorView(*this);
        }

        return TensorView(std::static_pointer_cast<ITensor>(m_tensor->as_type(dtype)));
    }

    TensorView tensor_like(const DType& dtype,
                           std::vector<neo::TensorIndex> shape,
                           std::vector<neo::TensorIndex> strides = {},
                           size_t offset                         = 0)
    {
        return TensorView(m_tensor->tensor_like(dtype, shape, strides, offset));
    }

  protected:
    std::shared_ptr<ITensor> get_tensor() const
    {
        return m_tensor;
    }

    void throw_on_invalid_storage();

  private:
    std::shared_ptr<ITensor> m_tensor;
};

// template <typename TensorT>
// struct ITensorOperationsPolicy
// {
//     static std::shared_ptr<TensorT> reshape(TensorT& self, const std::vector<neo::TensorIndex>& dims) = 0;
// };

// template <>
// struct ITensorOperationsPolicy<RMMTensor>
// {
//     static std::shared_ptr<RMMTensor> reshape(RMMTensor& self, const std::vector<neo::TensorIndex>& dims)
//     {
//         return std::make_shared<RMMTensor>();
//     }
// };

template <typename TensorT>
struct ITensorOperations
{
    std::shared_ptr<TensorT> reshape(const std::vector<neo::TensorIndex>& dims) const
    {
        return TensorAdaptor<TensorT>::reshape(*(TensorT*)this, dims);
    }
};

// template <typename TensorT>
// struct ITensorStorage
// {
//     virtual std::shared_ptr<TensorT> reshape(const std::vector<neo::TensorIndex>& dims) const = 0;
// };

template <typename TensorT>
struct ITensorObject : public ITensorOperations<TensorT>, public ITensor
{
    std::shared_ptr<ITensor> tensor_like(const DType& dtype,
                                         std::vector<neo::TensorIndex> shape,
                                         std::vector<neo::TensorIndex> strides = {},
                                         size_t offset                         = 0) const override
    {
        return TensorAdaptor<TensorT>::tensor_like(*(TensorT*)this, dtype, shape, strides, offset);
    }

    bool is_host_accessible() const override
    {
        return TensorAdaptor<TensorT>::IsHostAccessible;
    }

    bool is_device_accessible() const override
    {
        return TensorAdaptor<TensorT>::IsHostAccessible;
    }
};

struct IDeviceTensorObject
{
    virtual cudaStream_t stream() const = 0;
};

struct IHostTensorObject
{
    virtual std::vector<uint8_t> get_host_data() const = 0;
};

struct DeviceTensorView : public TensorView
{
    DeviceTensorView(std::shared_ptr<ITensor> tensor) : TensorView(tensor)
    {
        CHECK(this->get_tensor()->is_device_accessible()) << "Provided tensor is not device accessible";
    }

    DeviceTensorView(const TensorView& other) : TensorView(other)
    {
        CHECK(this->get_tensor()->is_device_accessible()) << "Provided tensor is not device accessible";
    }

    cudaStream_t stream() const
    {
        return std::static_pointer_cast<IDeviceTensorObject>(this->get_tensor())->stream();
    }
};

struct HostTensorView : public TensorView
{
    HostTensorView(std::shared_ptr<ITensor> tensor) : TensorView(tensor)
    {
        CHECK(this->get_tensor()->is_device_accessible()) << "Provided tensor is not device accessible";
    }

    HostTensorView(const TensorView& other) : TensorView(other)
    {
        CHECK(this->get_tensor()->is_host_accessible()) << "Provided tensor is not device accessible";
    }

    std::vector<uint8_t> get_host_data() const
    {
        return std::static_pointer_cast<IHostTensorObject>(this->get_tensor())->get_host_data();
    }
};

struct RMMTensor : public ITensorObject<RMMTensor>, IDeviceTensorObject, IHostTensorObject
{
  public:
    RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
              DType dtype,
              std::vector<neo::TensorIndex> shape,
              std::vector<neo::TensorIndex> stride = {},
              size_t offset                        = 0) :
      m_md(std::move(device_buffer)),
      m_offset(offset),
      m_dtype(std::move(dtype)),
      m_shape(std::move(shape)),
      m_stride(std::move(stride))
    {
        if (m_stride.empty())
        {
            trtlab::neo::detail::validate_stride(this->m_shape, this->m_stride);
        }

        DCHECK(m_offset + this->bytes() <= m_md->size())
            << "Inconsistent tensor. Tensor values would extend past the end of the device_buffer";
    }
    ~RMMTensor() = default;

    std::shared_ptr<neo::MemoryDescriptor> get_memory() const override
    {
        return nullptr;
    }

    void* data() const override
    {
        return static_cast<uint8_t*>(m_md->data()) + this->offset_bytes();
    }

    neo::RankType rank() const final
    {
        return m_shape.size();
    }

    trtlab::neo::DataType dtype() const override
    {
        return m_dtype;
    }

    std::size_t count() const final
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
    }

    std::size_t bytes() const final
    {
        return count() * m_dtype.item_size();
    }

    neo::TensorIndex shape(std::uint32_t idx) const final
    {
        DCHECK_LT(idx, m_shape.size());
        return m_shape.at(idx);
    }

    neo::TensorIndex stride(std::uint32_t idx) const final
    {
        DCHECK_LT(idx, m_stride.size());
        return m_stride.at(idx);
    }

    void get_shape(std::vector<neo::TensorIndex>& s) const final
    {
        s.resize(rank());
        std::copy(m_shape.begin(), m_shape.end(), s.begin());
    }

    void get_stride(std::vector<neo::TensorIndex>& s) const final
    {
        s.resize(rank());
        std::copy(m_stride.begin(), m_stride.end(), s.begin());
    }

    bool is_compact() const final
    {
        neo::TensorIndex ttl = 1;
        for (int i = rank() - 1; i >= 0; i--)
        {
            if (stride(i) != ttl)
            {
                return false;
            }

            ttl *= shape(i);
        }
        return true;
    }

    std::shared_ptr<neo::ITensor> slice(const std::vector<neo::TensorIndex>& min_dims,
                                        const std::vector<neo::TensorIndex>& max_dims) const override
    {
        // Calc new offset
        size_t offset = std::transform_reduce(
            m_stride.begin(), m_stride.end(), min_dims.begin(), m_offset, std::plus<>(), std::multiplies<>());

        // Calc new shape
        std::vector<neo::TensorIndex> shape;
        std::transform(max_dims.begin(), max_dims.end(), min_dims.begin(), std::back_inserter(shape), std::minus<>());

        // Stride remains the same

        return std::make_shared<RMMTensor>(m_md, offset, m_dtype, shape, m_stride);
    }

    std::shared_ptr<neo::ITensor> reshape(const std::vector<neo::TensorIndex>& dims) const override
    {
        return std::make_shared<RMMTensor>(m_md, 0, m_dtype, dims, m_stride);
    }

    std::shared_ptr<neo::ITensor> deep_copy() const override
    {
        // Deep copy
        std::shared_ptr<rmm::device_buffer> copied_buffer =
            std::make_shared<rmm::device_buffer>(*m_md, m_md->stream(), m_md->memory_resource());

        return std::make_shared<RMMTensor>(copied_buffer, m_offset, m_dtype, m_shape, m_stride);
    }

    // Tensor reshape(std::vector<neo::TensorIndex> shape)
    // {
    //     CHECK(is_compact());
    //     return Tensor(descriptor_shared(), dtype_size(), shape);
    // }

    std::shared_ptr<neo::ITensor> as_type(neo::DataType dtype) const override
    {
        DType new_dtype(dtype.type_id());

        auto input_type  = m_dtype.type_id();
        auto output_type = new_dtype.type_id();

        // Now do the conversion
        auto new_data_buffer = cast(DevMemInfo{this->count(), input_type, m_md, this->offset_bytes()}, output_type);

        // Return the new type
        return std::make_shared<RMMTensor>(new_data_buffer, 0, new_dtype, m_shape, m_stride);
    }

    cudaStream_t stream() const override
    {
        return m_md->stream().value();
    }

    std::vector<uint8_t> get_host_data() const override
    {
        std::vector<uint8_t> out_data;

        out_data.resize(this->bytes());

        NEO_CHECK_CUDA(cudaMemcpy(&out_data[0], this->data(), this->bytes(), cudaMemcpyDeviceToHost));

        return out_data;
    }

    static DeviceTensorView create(std::shared_ptr<rmm::device_buffer> device_buffer,
                                   DType dtype,
                                   std::vector<neo::TensorIndex> shape,
                                   std::vector<neo::TensorIndex> stride = {},
                                   size_t offset                        = 0)
    {
        auto tensor = std::make_shared<RMMTensor>(device_buffer, dtype, shape, stride, offset);

        return DeviceTensorView(tensor);
    }

  protected:
  private:
    size_t offset_bytes() const
    {
        return m_offset * m_dtype.item_size();
    }

    // Memory info
    std::shared_ptr<rmm::device_buffer> m_md;
    size_t m_offset;

    // // Type info
    // std::string m_typestr;
    // std::size_t m_dtype_size;
    DType m_dtype;

    // Shape info
    std::vector<neo::TensorIndex> m_shape;
    std::vector<neo::TensorIndex> m_stride;

    friend TensorAdaptor<RMMTensor>;
};

template <>
struct TensorAdaptor<RMMTensor>
{
    static constexpr bool IsHostAccessible   = true;
    static constexpr bool IsDeviceAccessible = true;

    static std::shared_ptr<RMMTensor> reshape(const RMMTensor& self, const std::vector<neo::TensorIndex>& dims)
    {
        return std::make_shared<RMMTensor>(self.m_md, self.dtype(), dims, self.m_stride, self.m_offset);
    }

    static std::shared_ptr<RMMTensor> tensor_like(const RMMTensor& self,
                                                  const DType& dtype,
                                                  std::vector<neo::TensorIndex> shape,
                                                  std::vector<neo::TensorIndex> strides,
                                                  size_t offset = 0)
    {
        size_t new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) * dtype.item_size();

        auto new_buffer =
            std::make_shared<rmm::device_buffer>(new_size, self.m_md->stream(), self.m_md->memory_resource());

        return std::make_shared<RMMTensor>(new_buffer, dtype, shape, strides, offset);
    }
};

// Before using this, cupy must be loaded into the module with `pyneo::import(m, "cupy")`
py::object tensor_to_cupy(const neo::TensorObject& tensor, const py::module_& mod)
{
    // These steps follow the cupy._convert_object_with_cuda_array_interface function shown here:
    // https://github.com/cupy/cupy/blob/a5b24f91d4d77fa03e6a4dd2ac954ff9a04e21f4/cupy/core/core.pyx#L2478-L2514
    auto cp      = mod.attr("cupy");
    auto cuda    = cp.attr("cuda");
    auto ndarray = cp.attr("ndarray");

    auto py_tensor = py::cast(tensor);

    auto ptr    = (uintptr_t)tensor.data();
    auto nbytes = tensor.bytes();
    auto owner  = py_tensor;
    int dev_id  = -1;

    py::list shape_list;
    py::list stride_list;

    for (auto& idx : tensor.get_shape())
    {
        shape_list.append(idx);
    }

    for (auto& idx : tensor.get_stride())
    {
        stride_list.append(idx * tensor.dtype_size());
    }

    py::object mem    = cuda.attr("UnownedMemory")(ptr, nbytes, owner, dev_id);
    py::object dtype  = cp.attr("dtype")(tensor.get_numpy_typestr());
    py::object memptr = cuda.attr("MemoryPointer")(mem, 0);

    // TODO(MDD): Sync on stream

    return ndarray(py::cast<py::tuple>(shape_list), dtype, memptr, py::cast<py::tuple>(stride_list));
}

DeviceTensorView cupy_to_tensor(py::object cupy_array)
{
    // Convert inputs from cupy to Tensor
    py::dict arr_interface = cupy_array.attr("__cuda_array_interface__");

    py::tuple shape_tup = arr_interface["shape"];

    auto shape = shape_tup.cast<std::vector<neo::TensorIndex>>();

    std::string typestr = arr_interface["typestr"].cast<std::string>();

    py::tuple data_tup = arr_interface["data"];

    uintptr_t data_ptr = data_tup[0].cast<uintptr_t>();

    std::vector<neo::TensorIndex> strides{};

    if (arr_interface.contains("strides") && !arr_interface["strides"].is_none())
    {
        py::tuple strides_tup = arr_interface["strides"];

        strides = strides_tup.cast<std::vector<neo::TensorIndex>>();
    }

    //  Get the size finally
    auto size = cupy_array.attr("data").attr("mem").attr("size").cast<size_t>();

    auto tensor = RMMTensor::create(
        std::make_shared<rmm::device_buffer>((void const*)data_ptr, size, rmm::cuda_stream_per_thread),
        DType::from_numpy(typestr),
        shape,
        strides,
        0);

    return tensor;
}

}  // namespace morpheus
