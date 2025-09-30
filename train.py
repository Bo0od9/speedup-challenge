import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_dataset, create_dataloaders
from model import Trompt

from profiling import ThroughputMeter, make_profiler, print_top_tables, marked, export_trace

TRAIN_URL = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
VAL_URL = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

# LOGDIR = "./profiler_logs"
LOGDIR = "/kaggle/working/profiler_logs"
PROFILE_STEPS = 50


def train_one_epoch(model, train_dl, optimizer, device, profile_first_epoch=True):
    model.train()
    meter = ThroughputMeter(drop_first_steps=20)

    if profile_first_epoch:
        prof, total_profiled = make_profiler(LOGDIR, wait=5, warmup=10, active=PROFILE_STEPS)
        with prof:
            for step, (x, y) in enumerate(train_dl, 1):
                bs = x.size(0)
                with marked("train_step"):
                    optimizer.zero_grad(set_to_none=True)
                    with marked("forward"):
                        pred = model(x.to(device))
                    with marked("loss"):
                        loss = F.mse_loss(pred, y.unsqueeze(1).repeat(1, len(model.tcells)).to(device))
                    with marked("backward"):
                        loss.backward()
                    with marked("optim_step"):
                        optimizer.step()

                meter.update(bs)
                prof.step()

                if step >= total_profiled:
                    break
        print_top_tables(prof, row_limit=40)
        export_trace(prof)
        print(f"\n[Throughput during profiled window] ~{meter.rate():.1f} samples/sec\n")

    else:
        for x, y in tqdm(train_dl):
            bs = x.size(0)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device))
            loss = F.mse_loss(pred, y.unsqueeze(1).repeat(1, len(model.tcells)).to(device))
            loss.backward()
            optimizer.step()
            meter.update(bs)

        print(f"\n[Throughput epoch] ~{meter.rate():.1f} samples/sec\n")


def main():
    torch.manual_seed(0)
    train_dataset, val_dataset, Y_mean, Y_std = load_dataset(TRAIN_URL, VAL_URL, cache_dir="./data")
    train_dl, val_dl = create_dataloaders(
        train_dataset, val_dataset,
        batch_size_train=1024, batch_size_val=2048,
        num_workers=4,
    )

    n_columns = train_dataset.tensors[0].shape[1]
    model = Trompt(n_columns=n_columns, n_prompts=128, d_model=128, n_cycles=6)
    model = torch.compile(
        model,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False
)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    EPOCHS = 5

    train_one_epoch(model, train_dl, optimizer, device, profile_first_epoch=False)


if __name__ == "__main__":
    main()
