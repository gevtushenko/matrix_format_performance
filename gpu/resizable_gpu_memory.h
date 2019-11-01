//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H
#define MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H

#include <memory>

void internal_resize (char *& data, size_t new_size);
void internal_free (char *& data);

template <typename T>
class resizable_gpu_memory
{
public:
  resizable_gpu_memory () = default;
  ~resizable_gpu_memory () { internal_free (data); }

  resizable_gpu_memory (const resizable_gpu_memory &) = delete;
  resizable_gpu_memory &operator= (const resizable_gpu_memory &) = delete;

  void clear ()
  {
    internal_free (data);
  }

  void resize (size_t new_size)
  {
    if (new_size > size)
    {
      size = new_size;
      internal_resize (data, size * sizeof (T));
    }
  }

  T *get () { return reinterpret_cast<T*> (data); }
  const T *get () const { return reinterpret_cast<const T*> (data); }

private:
  size_t size = 0;
  char *data = nullptr;
};

#endif // MATRIX_FORMAT_PERFORMANCE_RESIZABLE_GPU_MEMORY_H
