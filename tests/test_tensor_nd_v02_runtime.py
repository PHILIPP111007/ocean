from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
C_SOURCE = '\n#include <stdio.h>\n#include <stdlib.h>\n#include "std/tensor/tensor_runtime.h"\n#include "std/tensor/autograd_runtime.h"\nstatic void ck(int c,const char*m){if(!c){fprintf(stderr,"FAIL: %s\\n",m);exit(1);}}\nint main(void){\n size_t as[4]={2,2,2,3},bs[4]={1,2,3,4};\n ocean_tensor_handle_t a=ocean_tensor_zeros_nd(as,4,"float32","cpu"),b=ocean_tensor_zeros_nd(bs,4,"float32","cpu");\n for(size_t i=0;i<ocean_tensor_size(a);++i){size_t q[4],x=i;for(size_t j=4;j-- >0;){q[j]=x%as[j];x/=as[j];}ocean_tensor_set_nd_f32(a,q,4,(float)(i%7+1));}\n for(size_t i=0;i<ocean_tensor_size(b);++i){size_t q[4],x=i;for(size_t j=4;j-- >0;){q[j]=x%bs[j];x/=bs[j];}ocean_tensor_set_nd_f32(b,q,4,(float)(i%5+1));}\n ocean_tensor_handle_t c=ocean_tensor_matmul(a,b);ck(ocean_tensor_ndim(c)==4,"rank");ck(ocean_tensor_shape(c,0)==2&&ocean_tensor_shape(c,1)==2&&ocean_tensor_shape(c,2)==2&&ocean_tensor_shape(c,3)==4,"shape");\n ocean_tensor_handle_t t=ocean_tensor_transpose_dims(c,-2,-1);ck(ocean_tensor_shape(t,2)==4&&ocean_tensor_shape(t,3)==2,"transpose");\n ocean_tensor_handle_t s=ocean_tensor_sum_dim(c,-1,true);ck(ocean_tensor_shape(s,3)==1,"sum keepdim");\n ocean_autograd_set_requires_grad(a,true);ocean_autograd_set_requires_grad(b,true);\n ocean_tensor_handle_t y=ocean_autograd_matmul(a,b);size_t ys[4]={2,2,2,4};ocean_tensor_handle_t z=ocean_tensor_zeros_nd(ys,4,"float32","cpu");ocean_tensor_handle_t loss=ocean_autograd_mse_loss(y,z);ocean_autograd_backward(loss);\n ck(ocean_autograd_has_grad(a),"grad a");ck(ocean_autograd_has_grad(b),"grad b");ocean_tensor_handle_t gb=ocean_autograd_grad_copy(b);ck(ocean_tensor_shape(gb,0)==1,"broadcast grad reduced");\n ocean_tensor_release(gb);ocean_tensor_release(loss);ocean_tensor_release(z);ocean_tensor_release(y);ocean_tensor_release(s);ocean_tensor_release(t);ocean_tensor_release(c);ocean_tensor_release(b);ocean_tensor_release(a);puts("ND Tensor/autograd v0.2: OK");return 0;\n}\n'

def test_nd_tensor_autograd_v02_runtime():
    with tempfile.TemporaryDirectory(prefix='ocean_nd_v02_test_') as td:
        td = Path(td)
        src = td / 'test.c'
        binary = td / 'test'
        src.write_text(C_SOURCE, encoding='utf-8')
        subprocess.run([
            'gcc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
            '-I', str(ROOT), str(src),
            str(ROOT / 'std/tensor/autograd_runtime.c'),
            str(ROOT / 'std/tensor/tensor_runtime.c'),
            '-lm', '-o', str(binary),
        ], check=True)
        completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        assert 'ND Tensor/autograd v0.2: OK' in completed.stdout
