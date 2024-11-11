import os
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from dbg_model import Transformer, ModelArgs

args = ModelArgs()
bs = 256
seqlen = 128

device = torch.device("cuda")

x = torch.randint(low=0, high=args.vocab_size, size=(bs, seqlen)).to(device)
heads = torch.randint(low=0, high=args.n_output_heads, size=(bs,)).to(device)

model = Transformer(args)
model.to(device)
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
torch.cuda.synchronize()
optimizer = torch.optim.SGD(lr=0.000001, params=model.parameters())

step_flops = 6 * model_size * bs * seqlen
total_flops = 100 * 10**12
steps = int(total_flops / step_flops)
burnin_steps = steps // 2
assert steps > burnin_steps
print(f"{steps} steps with batch size ({bs},{seqlen})")
print(f"Model_size: {model_size:.2e}")
print(f"One step is {step_flops:.2e} FLOPs")
save_prof = None
for i in range(steps):
    if i == burnin_steps:
        after_burnin = time.time()

    def compute_loss():
        out = model(x, 0, heads)
        loss = (out**2).mean()
        return loss

    def train():
        optimizer.zero_grad()
        if os.environ.get("PROFILE") == "1" and i == burnin_steps:
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                with record_function("inference"):
                    loss = compute_loss()
                with record_function("backward"):
                    loss.backward()
                with record_function("optimizer"):
                    optimizer.step()
        else:
            loss = compute_loss()
            loss.backward()
            prof = None
            optimizer.step()
        return prof

    prof = train()
    torch.cuda.synchronize()
    if i == burnin_steps and prof:
        save_prof = prof

time_took = time.time() - after_burnin
flops = 6 * model_size * bs * seqlen * (steps - burnin_steps)
print(f"{time_took} seconds")
print(f"{flops / time_took / 10 ** 12} teraFLOPs per second")

if save_prof:
    events = [
        event
        for event in save_prof.key_averages()
        if hasattr(event, "device_time_total")
    ]
    events = list(reversed(sorted(events, key=lambda k: k.device_time_total)))
    for event in events[:10]:
        print(f"{event.key}:")
        print(f"\t{int(event.device_time_total/1000):.1f}ms, {event.count}")
