// Reusable temporary storage for Thrust code that runs repeatedly (e.g. collision detection and
// the collision benchmark's oracle). Include it from translation units compiled for the active
// Thrust backend, after any THRUST_*_SYSTEM defines.
#pragma once

#include <cstddef>
#include <map>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace happyray
{

	// Caching allocator for Thrust's temporary storage, passed through the execution policy
	// (`thrust::device(allocator)`): freed blocks are kept and handed out again for requests they
	// can hold, so repeated calls stop allocating once they have seen their largest working set.
	// Not thread-safe.
	class CachingAllocator
	{
	  public:
		using value_type = char;

		CachingAllocator()									  = default;
		CachingAllocator(const CachingAllocator &)			  = delete;
		CachingAllocator &operator=(const CachingAllocator &) = delete;

		~CachingAllocator()
		{
			for (const auto &[size, block] : free_blocks_)
				thrust::device_free(thrust::device_pointer_cast(block));
			for (const auto &[block, size] : used_blocks_)
				thrust::device_free(thrust::device_pointer_cast(block));
		}

		char *allocate(std::ptrdiff_t bytes)
		{
			const size_t size = static_cast<size_t>(bytes);
			auto		 it	  = free_blocks_.lower_bound(size);
			char		*block;
			size_t		 block_size;
			if (it != free_blocks_.end())
			{
				block_size = it->first;
				block	   = it->second;
				free_blocks_.erase(it);
			}
			else
			{
				block_size = size;
				block	   = thrust::raw_pointer_cast(thrust::device_malloc<char>(size));
			}
			used_blocks_.emplace(block, block_size);
			return block;
		}

		void deallocate(char *block, size_t)
		{
			const auto it = used_blocks_.find(block);
			free_blocks_.emplace(it->second, block);
			used_blocks_.erase(it);
		}

	  private:
		std::multimap<size_t, char *> free_blocks_; // By size, for best-fit reuse.
		std::map<char *, size_t>	  used_blocks_;
	};

	// Grows `values` to at least `count` elements; never shrinks, so the storage persists.
	// Callers address their first `count` elements explicitly.
	template <typename T> T *Reserve(thrust::device_vector<T> *values, size_t count)
	{
		if (values->size() < count)
		{
			// Drop the old storage first: its contents are scratch and need no copying.
			values->clear();
			values->shrink_to_fit();
			values->resize(count);
		}
		return thrust::raw_pointer_cast(values->data());
	}

} // namespace happyray
