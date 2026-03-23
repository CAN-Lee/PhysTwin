"""Minimal test: does Warp's adjoint work for array writes through structs?"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import warp as wp
import torch
wp.init()

@wp.struct
class State:
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)

@wp.kernel
def simple_update(state: State, scale: float):
    p = wp.tid()
    state.x[p] = state.x[p] + state.v[p] * scale

@wp.kernel
def simple_update_no_struct(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), scale: float):
    p = wp.tid()
    x[p] = x[p] + v[p] * scale

device = 'cuda:0'
N = 10

# TEST 1: With struct
print("=== TEST 1: Struct-based kernel ===")
s = State()
s.x = wp.from_torch(torch.full((N,3), 0.5, device=device), dtype=wp.vec3, requires_grad=True)
s.v = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3, requires_grad=True)

tape = wp.Tape()
with tape:
    wp.launch(simple_update, N, [s, 0.1], device=device)

seed = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3)
tape.backward(grads={s.x: seed})

print(f"  tape.gradients count: {len(tape.gradients)}")
for name, arr in [('x', s.x), ('v', s.v)]:
    in_g = arr in tape.gradients
    if in_g:
        g = wp.to_torch(tape.gradients[arr])
        print(f"  {name}: norm={torch.norm(g).item():.6e}, sample={g[0].tolist()}")
    else:
        print(f"  {name}: NOT in gradients")

# TEST 2: Without struct (direct arrays)
print("\n=== TEST 2: Direct array kernel ===")
x2 = wp.from_torch(torch.full((N,3), 0.5, device=device), dtype=wp.vec3, requires_grad=True)
v2 = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3, requires_grad=True)

tape2 = wp.Tape()
with tape2:
    wp.launch(simple_update_no_struct, N, [x2, v2, 0.1], device=device)

seed2 = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3)
tape2.backward(grads={x2: seed2})

print(f"  tape.gradients count: {len(tape2.gradients)}")
for name, arr in [('x', x2), ('v', v2)]:
    in_g = arr in tape2.gradients
    if in_g:
        g = wp.to_torch(tape2.gradients[arr])
        print(f"  {name}: norm={torch.norm(g).item():.6e}, sample={g[0].tolist()}")
    else:
        print(f"  {name}: NOT in gradients")

# TEST 3: Struct + manual grad check
print("\n=== TEST 3: Struct with manual grad check after backward ===")
s3 = State()
s3.x = wp.from_torch(torch.full((N,3), 0.5, device=device), dtype=wp.vec3, requires_grad=True)
s3.v = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3, requires_grad=True)

tape3 = wp.Tape()
with tape3:
    wp.launch(simple_update, N, [s3, 0.1], device=device)

seed3 = wp.from_torch(torch.full((N,3), 1.0, device=device), dtype=wp.vec3)
tape3.backward(grads={s3.x: seed3})

print(f"  s3.v.grad: {wp.to_torch(s3.v.grad)[0].tolist() if s3.v.grad is not None else 'None'}")
print(f"  s3.x.grad: {wp.to_torch(s3.x.grad)[0].tolist() if s3.x.grad is not None else 'None'}")
