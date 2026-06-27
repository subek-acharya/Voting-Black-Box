# SMOOAttackExperiment_Unet_Cait.py

"""
SMOO (Sparse Multi-Objective Optimization Attack) Experiment - UNet + Model
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import EXPERIMENTS_ALL, UNET_CHECKPOINT
from AttackWrapperSMOO import SMOO_AttackWrapper
from ModelFactory import ModelFactory
import utils


class SMOOAttackExperiment:
    def __init__(
        self,
        experiments_config,
        sparsity_k=20,
        iterations=5000,
        pc=0.1,
        pm=0.4,
        pop_size=2,
        total_samples=500,
        n_classes=2,
    ):
        self.experiments = experiments_config
        self.sparsity_k = sparsity_k
        self.iterations = iterations
        self.pc = pc
        self.pm = pm
        self.pop_size = pop_size
        self.total_samples = total_samples
        self.n_classes = n_classes
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factory = ModelFactory(device=self.device)

    def _get_smoo_config(self, model_name, use_unet):
        unet_suffix = "_unet" if use_unet else ""
        return {
            "eps": self.sparsity_k,
            "iterations": self.iterations,
            "pc": self.pc,
            "pm": self.pm,
            "pop_size": self.pop_size,
            "save_directory": f"./smoo_results/{model_name}{unet_suffix}_k={self.sparsity_k}",
        }

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            use_unet = config.get("use_unet", False)
            unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
            mode_str = "UNet + Model" if use_unet else "Model Only"

            print(f"\n{'=' * 70}")
            print(f"Model: {model_name} ({mode_str})")
            print(f"Dataset: {config['dataset_path']}")
            print(f"Sparsity (k): {self.sparsity_k}, Query Limit: {self.iterations}")
            print(f"Parameters: pc={self.pc}, pm={self.pm}, pop_size={self.pop_size}")
            print(f"Total Samples: {self.total_samples}")
            print(f"{'=' * 70}")

            model = self._load_model(model_name, config, use_unet, unet_ckpt)

            loader = self._get_clean_loader(config["dataset_path"], model, batch_size)
            if not loader:
                self._append_result(model_name, None, use_unet)
                continue

            adv_loader, acc = self._execute_attack(model, loader, model_name, use_unet)
            
            unet_suffix = "_unet" if use_unet else ""
            result_key = f"{model_name}{unet_suffix}_k={self.sparsity_k}"
            self.results[result_key] = acc

            self._append_result(model_name, acc, use_unet)

            if adv_loader:
                self._save_samples(adv_loader, model_name, config["dataset_path"], use_unet)

    def _load_model(self, model_name, config, use_unet, unet_ckpt):
        """Load model with or without UNet wrapper."""
        if use_unet:
            model = self.factory.get_unet_model_wrapper(
                model_name=model_name,
                model_checkpoint=config["ckpt_path"],
                unet_checkpoint=unet_ckpt,
            )
        else:
            model = self.factory.get_model(model_name, config["ckpt_path"])
        
        model.to(self.device)
        model.eval()
        return model

    def _execute_attack(self, model, loader, model_name, use_unet):
        try:
            adv_loader = SMOO_AttackWrapper(
                model=model,
                device=self.device,
                dataLoader=loader,
                config=self._get_smoo_config(model_name, use_unet),
            )
            acc = utils.validateD(adv_loader, model, self.device)
            print(f"\nRobust Accuracy (k={self.sparsity_k}): {acc:.4f}")
            return adv_loader, acc
        except Exception as e:
            print(f"Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_clean_loader(self, dataset_path, model, batch_size):
        try:
            if not os.path.exists(dataset_path):
                print(f"ERROR: Dataset not found: {dataset_path}")
                return None
            
            data = torch.load(dataset_path, weights_only=False)
            
            # Handle different data formats (scanned vs original)
            if "xData" in data:
                images = data["xData"].float()
                labels = data["yDataBinary"].long()
            elif "data" in data:
                images = data["data"].float()
                labels = data["binary_labels"].long()
            else:
                raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")
            
            dataset = TensorDataset(images, labels)
            raw_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            return utils.GetCorrectlyIdentifiedSamplesBalanced(
                model, self.total_samples, raw_loader, numClasses=self.n_classes
            )
        except Exception as e:
            print(f"Dataloader failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_samples(self, adv_loader, model_name, dataset_path, use_unet, n_save=500):
        unet_suffix = "_unet" if use_unet else ""
        base_dir = os.path.join("adv_samples", f"smoo{unet_suffix}", model_name)
        os.makedirs(base_dir, exist_ok=True)
        
        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace(".pt", "").replace("val_", "")
        )
        filename = f"{model_name}{unet_suffix}_k={self.sparsity_k}_{clean_name}.pt"
        save_path = os.path.join(base_dir, filename)

        # Convert dataloader to tensors using utils
        x_adv, y_clean = utils.DataLoaderToTensor(adv_loader)
        
        # Limit to n_save samples
        x_adv = x_adv[:n_save]
        y_clean = y_clean[:n_save]

        # Save in target format
        output_data = {
            "data": x_adv.float(),
            "binary_labels": y_clean.long(),
            "original_labels": y_clean.long(),
        }
        
        torch.save(output_data, save_path)
        
        print(f"\n{'─' * 60}")
        print(f"Saved {len(x_adv)} samples to {save_path}")
        print(f"  data shape: {output_data['data'].shape}")
        print(f"  binary_labels: {torch.bincount(output_data['binary_labels']).tolist()}")
        print(f"{'─' * 60}")

    def _append_result(self, model_name, acc, use_unet, filepath=None):
        if filepath is None:
            filepath = "smoo_unet_results.txt" if use_unet else "smoo_results.txt"
        
        mode_str = "(UNet+Model)" if use_unet else ""
        unet_suffix = "_unet" if use_unet else ""
        
        with open(filepath, "a", encoding="utf-8") as f:
            header_key = (self.sparsity_k, use_unet)
            if header_key not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + f" SMOO ATTACK L0 {mode_str} FINAL RESULTS k={self.sparsity_k} "
                    + "=" * 10
                    + "\n"
                )
                self.headers_written.add(header_key)
            
            result_key = f"{model_name}{unet_suffix}_k={self.sparsity_k}"
            if acc is not None:
                f.write(f"{result_key:<40}: {acc:.4f}\n")
            else:
                f.write(f"{result_key:<40}: Failed\n")

def main():
    sparsity = [20, 10, 1]
    # sparsity = [10]
    for k in sparsity:
        experiment = SMOOAttackExperiment(
            experiments_config=EXPERIMENTS_ALL,
            sparsity_k=k,
            iterations=5000,
            pc=0.1,
            pm=0.4,
            pop_size=2,
            total_samples=500,
        )
        experiment.run_all()

if __name__ == "__main__":
    main()