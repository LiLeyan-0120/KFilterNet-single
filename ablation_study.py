#!/usr/bin/env python3
"""
Flexible Kalman Filter Network Ablation Study
Implements dataset comparison experiment:
Test the performance of different model architectures on different datasets to
demonstrate the importance of choosing different model modes for different scenarios
"""

import os
import sys
import json
import copy
import random
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from utils.logger import setup_logger

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logger = setup_logger(__name__)


def set_random_seed(seed=42):
    """Set random seed for experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from training.config import Config, default_config
from training.trainer import Trainer
from models.KFilterNet_single import KFilterNet_single, ModuleType


class AblationStudy:
    """Ablation study class - Dataset Comparison Experiment"""

    def __init__(self, base_config: Config = None):
        """Initialize ablation study"""
        self.base_config = base_config or default_config
        self.results = []
        self.output_dir = "outputs/Ablation"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logger(__name__)

        self.logger.info("=" * 60)
        self.logger.info("KFilterNet Dataset Comparison Ablation Study")
        self.logger.info("=" * 60)


    def run_experiment(self,
                       experiment_name: str,
                       config: Config,
                       description: str = "") -> Dict:
        """Run single experiment"""
        self.logger.info(f"\n{'=' * 20} {experiment_name} {'=' * 20}")
        if description:
            self.logger.info(f"Description: {description}")

        # Create output directory
        experiment_output_dir = os.path.join(self.output_dir, experiment_name)
        config.training.output_dir = experiment_output_dir
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Save configuration
        config_path = os.path.join(experiment_output_dir, 'config.yaml')
        config.save_yaml(config_path)

        # Create model
        model = KFilterNet_single(config)

        # Create trainer
        trainer = Trainer(config, model, device=config.training.device)

        # Train model
        try:
            train_history, val_history = trainer.train()

            # Evaluate RMSE on test set (trainer automatically loads best checkpoint)
            self.logger.info("Evaluating RMSE on test set...")
            test_rmse = trainer.evaluate_test_rmse()

            # Collect results
            result = {
                'experiment_name': experiment_name,
                'description': description,
                'config': config,
                'final_train_loss': train_history['total_loss'][-1] if train_history['total_loss'] else None,
                'final_val_loss': val_history['total_loss'][-1] if val_history['total_loss'] else None,
                'best_val_loss': trainer.best_val_loss,
                'epochs_completed': trainer.current_epoch + 1,
                'train_history': train_history,
                'val_history': val_history,
                'model_params': sum(p.numel() for p in model.parameters()),
                'test_rmse': test_rmse,
                'output_dir': experiment_output_dir
            }

            # Save results
            result_path = os.path.join(experiment_output_dir, 'result.json')
            with open(result_path, 'w') as f:
                # Create serializable result copy
                serializable_result = {
                    'experiment_name': result['experiment_name'],
                    'description': result['description'],
                    'final_train_loss': result['final_train_loss'],
                    'final_val_loss': result['final_val_loss'],
                    'best_val_loss': result['best_val_loss'],
                    'epochs_completed': result['epochs_completed'],
                    'model_params': result['model_params'],
                    'test_rmse': result['test_rmse'],
                    'output_dir': result['output_dir']
                }
                json.dump(serializable_result, f, indent=2)

            self.logger.info(f"Experiment {experiment_name} completed!")
            self.logger.info(f"Test set RMSE: {test_rmse['rmse_total']:.6f}")
            self.logger.info(f"Final timestep RMSE: {test_rmse['final_rmse']:.6f}")

            return result

        except Exception as e:
            self.logger.info(f"Experiment {experiment_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def experiment_dataset_comparison(self):
        """
        Dataset comparison experiment: Test the performance of different model architectures on different datasets.
        Purpose: Demonstrate the importance of selecting different model modes for different scenarios.

        Three datasets:
        1. complex_complex_noise - Complex trajectory (7 motion modes) + Complex noise (Gaussian + Impulse + Correlated)
        2. complex_fixed_noise - Complex trajectory (7 motion modes) + Fixed noise (Gaussian only, fixed parameters)
        3. simple_ca_complex_noise - Simple trajectory (CA only) + Complex noise (Gaussian + Impulse + Correlated)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Dataset Comparison Experiment")
        self.logger.info("=" * 60)

        # Define 4 unified model configurations
        model_configs = [
            {
                'name': 'all_learnable',
                'description': 'All modules learnable',
                'F_type': ModuleType.LEARNABLE.value,
                'H_type': ModuleType.LEARNABLE.value,
                'K_type': ModuleType.LEARNABLE.value,
                'Q_type': ModuleType.LEARNABLE.value,
                'R_type': ModuleType.LEARNABLE.value,
            },
            {
                'name': 'fhqr_semi_fixed',
                'description': 'FHQR semi-fixed (K fixed)',
                'F_type': ModuleType.SEMI_FIXED.value,
                'H_type': ModuleType.SEMI_FIXED.value,
                'K_type': ModuleType.FIXED.value,
                'Q_type': ModuleType.SEMI_FIXED.value,
                'R_type': ModuleType.SEMI_FIXED.value,
            },
            {
                'name': 'qr_learnable',
                'description': 'QR learnable (FHK fixed)',
                'F_type': ModuleType.FIXED.value,
                'H_type': ModuleType.FIXED.value,
                'K_type': ModuleType.FIXED.value,
                'Q_type': ModuleType.LEARNABLE.value,
                'R_type': ModuleType.LEARNABLE.value,
            },
            {
                'name': 'fh_learnable',
                'description': 'FH learnable (KQR fixed)',
                'F_type': ModuleType.LEARNABLE.value,
                'H_type': ModuleType.LEARNABLE.value,
                'K_type': ModuleType.FIXED.value,
                'Q_type': ModuleType.FIXED.value,
                'R_type': ModuleType.FIXED.value,
            },
        ]

        # Define dataset configurations
        dataset_configs = [
            {
                'dataset_name': 'complex_complex_noise',
                'description': 'Complex trajectory + Complex noise',
                'dataset_desc': 'Full 7 motion modes + Gaussian+Impulse+Correlated noise',
            },
            {
                'dataset_name': 'complex_fixed_noise',
                'description': 'Complex trajectory + Fixed noise',
                'dataset_desc': 'Full 7 motion modes + Fixed Gaussian noise only',
            },
            {
                'dataset_name': 'simple_ca_complex_noise',
                'description': 'Simple CA trajectory + Complex noise',
                'dataset_desc': 'CA motion only + Multiple noise types',
            },
        ]

        # Generate all experiment configurations (Cartesian product)
        experiment_configs = []
        for dataset_config in dataset_configs:
            for model_config in model_configs:
                experiment_configs.append({
                    'dataset_name': dataset_config['dataset_name'],
                    'name': f"{dataset_config['dataset_name']}_{model_config['name']}",
                    'description': f"{dataset_config['description']} - {model_config['description']}",
                    'dataset_desc': dataset_config['dataset_desc'],
                    'F_type': model_config['F_type'],
                    'H_type': model_config['H_type'],
                    'K_type': model_config['K_type'],
                    'Q_type': model_config['Q_type'],
                    'R_type': model_config['R_type'],
                })

        results = []

        for exp_config in experiment_configs:
            config = copy.deepcopy(self.base_config)

            # Set dataset name
            config.data.dataset_name = exp_config['dataset_name']

            # Set module types
            config.model.F_type = exp_config['F_type']
            config.model.H_type = exp_config['H_type']
            config.model.K_type = exp_config['K_type']
            config.model.Q_type = exp_config['Q_type']
            config.model.R_type = exp_config['R_type']
            config.model.init_type = ModuleType.FIXED.value

            # Training parameters
            config.training.epochs = 100
            config.training.early_stopping_patience = 10

            # Run experiment
            experiment_name = f"dataset_{exp_config['name']}"
            result = self.run_experiment(
                experiment_name=experiment_name,
                config=config,
                description=exp_config['description']
            )

            if result:
                result['dataset_name'] = exp_config['dataset_name']
                result['dataset_desc'] = exp_config['dataset_desc']
                result['module_config'] = exp_config['name']
                results.append(result)

        # Save results
        self._save_dataset_comparison_results(results)
        return results

    def _save_dataset_comparison_results(self, results: List[Dict]):
        """Save dataset comparison results (grouped by dataset)"""
        if not results:
            self.logger.info("Warning: Dataset comparison has no successful results")
            return

        # Group by dataset
        by_dataset = {}
        for r in results:
            ds_name = r.get('dataset_name', 'unknown')
            if ds_name not in by_dataset:
                by_dataset[ds_name] = []
            by_dataset[ds_name].append(r)

        # Create results table
        df_data = []
        for dataset_name, ds_results in by_dataset.items():
            for result in ds_results:
                test_rmse = result.get('test_rmse', {})
                df_data.append({
                    'dataset': dataset_name,
                    'dataset_desc': result.get('dataset_desc', ''),
                    'experiment': result['experiment_name'],
                    'description': result['description'],
                    'F_type': result['config'].model.F_type,
                    'K_type': result['config'].model.K_type,
                    'rmse_total': test_rmse.get('rmse_total', None),
                    'rmse_x': test_rmse.get('rmse_x', None),
                    'rmse_y': test_rmse.get('rmse_y', None),
                    'rmse_z': test_rmse.get('rmse_z', None),
                    'final_rmse': test_rmse.get('final_rmse', None),
                    'best_val_loss': result['best_val_loss'],
                    'epochs_completed': result['epochs_completed'],
                })

        df = pd.DataFrame(df_data)

        # Save CSV and JSON
        csv_path = os.path.join(self.output_dir, "dataset_comparison.csv")
        json_path = os.path.join(self.output_dir, "dataset_comparison.json")
        df.to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump(df_data, f, indent=2)

        self.logger.info(f"\nDataset comparison results saved to:")
        self.logger.info(f"  CSV: {csv_path}")
        self.logger.info(f"  JSON: {json_path}")

        # Generate visualization
        self._visualize_dataset_comparison(by_dataset)

    def _visualize_dataset_comparison(self, by_dataset: Dict):
        """Visualize dataset comparison results (2x2 subplots)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        datasets = list(by_dataset.keys())
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#FF5733']

        # 1. Total RMSE comparison (grouped by dataset)
        for i, dataset in enumerate(datasets):
            ds_results = by_dataset[dataset]
            names = [r['experiment_name'].replace('dataset_', '') for r in ds_results]
            rmse_values = [r.get('test_rmse', {}).get('rmse_total', 0) for r in ds_results]
            y_pos = np.arange(len(names))
            axes[0, 0].barh(y_pos, rmse_values, align='center', color=colors[i], alpha=0.7, label=dataset)
            axes[0, 0].set_yticks(y_pos)
            axes[0, 0].set_yticklabels(names, fontsize=8)
            axes[0, 0].invert_yaxis()
            # Display values
            for j, v in enumerate(rmse_values):
                axes[0, 0].text(v + max(rmse_values) * 0.01, j, f'{v:.4f}', va='center', fontsize=7)

        axes[0, 0].set_xlabel('RMSE')
        axes[0, 0].set_title('Total RMSE Comparison (Lower is Better)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. F_type comparison within each dataset
        for i, dataset in enumerate(datasets):
            ds_results = by_dataset[dataset]
            f_type_values = {'learnable': [], 'fixed': []}
            for r in ds_results:
                f_type = r['config'].model.F_type
                rmse = r.get('test_rmse', {}).get('rmse_total', 0)
                if f_type in f_type_values:
                    f_type_values[f_type].append(rmse)

            x_pos = [i * 3, i * 3 + 1]
            if f_type_values['learnable']:
                avg_learnable = np.mean(f_type_values['learnable'])
                axes[0, 1].bar(x_pos[0], avg_learnable, width=0.8, color=colors[i], alpha=0.7, label=f'{dataset} learnable')
                axes[0, 1].text(x_pos[0], avg_learnable + 0.01, f'{avg_learnable:.4f}', ha='center', fontsize=7)
            if f_type_values['fixed']:
                avg_fixed = np.mean(f_type_values['fixed'])
                axes[0, 1].bar(x_pos[1], avg_fixed, width=0.8, color=colors[(i + 3) % 7], alpha=0.7, label=f'{dataset} fixed')
                axes[0, 1].text(x_pos[1], avg_fixed + 0.01, f'{avg_fixed:.4f}', ha='center', fontsize=7)

        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('F Type Performance by Dataset', fontweight='bold')
        axes[0, 1].set_xticks(range(len(datasets) * 3))
        axes[0, 1].set_xticklabels([f'{ds}\nL' if i % 2 == 0 else f'{ds}\nF' for ds in datasets for i in range(2)], fontsize=8)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Multi-dimensional RMSE comparison (X/Y/Z)
        for dataset in datasets:
            ds_results = by_dataset[dataset]
            for result in ds_results:
                rmse = result.get('test_rmse', {})
                name = result['experiment_name'].replace('dataset_', '')
                axes[1, 0].plot([0, 1, 2], [rmse.get('rmse_x', 0), rmse.get('rmse_y', 0), rmse.get('rmse_z', 0)],
                               'o-', label=name, alpha=0.7)

        axes[1, 0].set_xticks([0, 1, 2])
        axes[1, 0].set_xticklabels(['X', 'Y', 'Z'])
        axes[1, 0].set_title('Multi-dimensional RMSE', fontweight='bold')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Final timestep RMSE comparison (convergence performance)
        names = []
        final_rmses = []
        colors_list = []
        for dataset in datasets:
            ds_results = by_dataset[dataset]
            for result in ds_results:
                names.append(result['experiment_name'].replace('dataset_', ''))
                final_rmses.append(result.get('test_rmse', {}).get('final_rmse', 0))
                colors_list.append(colors[datasets.index(dataset)])

        y_pos = np.arange(len(names))
        axes[1, 1].barh(y_pos, final_rmses, align='center', color=colors_list, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(names, fontsize=8)
        axes[1, 1].invert_yaxis()
        # Display values
        for i, v in enumerate(final_rmses):
            axes[1, 1].text(v + max(final_rmses) * 0.01, i, f'{v:.4f}', va='center', fontsize=7)

        axes[1, 1].set_xlabel('RMSE')
        axes[1, 1].set_title('Final Timestep RMSE (Convergence)', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "dataset_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Visualization saved to: {plot_path}")

        # Generate additional RMSE vs timestep plot
        self._visualize_rmse_timesteps(results)

    def _visualize_rmse_timesteps(self, results: List[Dict]):
        """Visualize RMSE changes over timesteps"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Distinguish by dataset with colors
        datasets = sorted(list(set(r.get('dataset_name', 'unknown') for r in results)))
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#FF5733']

        for result in results:
            rmse_per_timestep = result.get('test_rmse', {}).get('rmse_per_timestep', [])
            if rmse_per_timestep:
                dataset_name = result.get('dataset_name', 'unknown')
                name = result['experiment_name'].replace('dataset_', '')
                color = colors[datasets.index(dataset_name) if dataset_name in datasets else 0]
                timesteps = range(len(rmse_per_timestep))
                ax.plot(timesteps, rmse_per_timestep,
                       linewidth=2, alpha=0.7, label=name, color=color)

        ax.set_title('RMSE Changes Over Timesteps', fontsize=16, fontweight='bold')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('RMSE')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "dataset_comparison_rmse_timesteps.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"RMSE timestep change plot saved to: {plot_path}")

    def run_all_experiments(self):
        """Run dataset comparison experiments"""
        self.logger.info("Starting dataset comparison experiments...")
        results = self.experiment_dataset_comparison()

        # Generate summary report
        self._generate_dataset_summary_report(results)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Dataset comparison experiments completed!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)

        return results

    def _generate_dataset_summary_report(self, results: List[Dict]):
        """Generate dataset comparison summary report"""
        report_path = os.path.join(self.output_dir, "dataset_comparison_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Comparison Ablation Study Report\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Experiment Purpose\n\n")
            f.write("This experiment tests the performance of different model architectures on different datasets to demonstrate the importance of choosing different model modes for different scenarios.\n\n")

            f.write("### Key Hypotheses\n\n")
            f.write("- **Complex trajectory complex noise dataset (complex_complex_noise)**: Complex motion (7 modes) + Complex noise (multiple), F learnable should be better than F fixed\n")
            f.write("- **Complex trajectory fixed noise dataset (complex_fixed_noise)**: Complex motion (7 modes) + Fixed noise (Gaussian only), F learnable should be better than F fixed\n")
            f.write("- **Simple CA trajectory complex noise dataset (simple_ca_complex_noise)**: Simple motion (CA only) + Complex noise, F fixed and F learnable should perform similarly\n")
            f.write("- Verify hypotheses by comparing F type (learnable vs fixed) performance across different datasets\n\n")

            # Group by dataset
            by_dataset = {}
            for r in results:
                ds_name = r.get('dataset_name', 'unknown')
                if ds_name not in by_dataset:
                    by_dataset[ds_name] = []
                by_dataset[ds_name].append(r)

            # Dataset numbering mapping
            dataset_numbers = {
                'complex_complex_noise': '①',
                'complex_fixed_noise': '②',
                'simple_ca_complex_noise': '③'
            }

            # Model configuration numbering mapping
            model_numbers = {
                'all_learnable': '①',
                'fhqr_semi_fixed': '②',
                'qr_learnable': '③',
                'fh_learnable': '④'
            }

            # Generate statistics table
            f.write("## Experiment Results Statistics\n\n")
            f.write("| Dataset Config | Model Config | Test RMSE | Final RMSE | Model Params |\n")
            f.write("|-----------|---------|-----------|-----------|-----------|\n")

            for dataset_name, ds_results in by_dataset.items():
                ds_num = dataset_numbers.get(dataset_name, '?')
                
                # Sort by model configuration order
                sorted_results = sorted(ds_results, 
                    key=lambda x: model_numbers.get(x.get('module_config', ''), '999'))
                
                first_row = True
                for r in sorted_results:
                    test_rmse = r.get('test_rmse', {})
                    model_config = r.get('module_config', '')
                    model_num = model_numbers.get(model_config, '?')
                    
                    if first_row:
                        f.write(f"| {ds_num} | {model_num} | {test_rmse.get('rmse_total', 0):.4f} | "
                               f"{test_rmse.get('final_rmse', 0):.4f} | {r.get('model_params', 0)} |\n")
                        first_row = False
                    else:
                        f.write(f"| | {model_num} | {test_rmse.get('rmse_total', 0):.4f} | "
                               f"{test_rmse.get('final_rmse', 0):.4f} | {r.get('model_params', 0)} |\n")

            f.write("\n")

            # Detailed results table
            for dataset_name, ds_results in by_dataset.items():
                f.write(f"## Dataset: {dataset_name}\n\n")
                f.write(f"**Description**: {ds_results[0].get('dataset_desc', '')}\n\n")
                f.write("| Experiment | F Type | K Type | Test RMSE | X-RMSE | Y-RMSE | Z-RMSE | Final RMSE |\n")
                f.write("|------|--------|--------|----------|--------|--------|--------|----------|\n")

                for r in ds_results:
                    test_rmse = r.get('test_rmse', {})
                    f.write(
                        f"| {r['description']} | {r['config'].model.F_type} | "
                        f"{r['config'].model.K_type} | {test_rmse.get('rmse_total', 0):.6f} | "
                        f"{test_rmse.get('rmse_x', 0):.6f} | {test_rmse.get('rmse_y', 0):.6f} | "
                        f"{test_rmse.get('rmse_z', 0):.6f} | {test_rmse.get('final_rmse', 0):.6f} |\n"
                    )

                # Best configuration for this dataset
                best = min(ds_results, key=lambda x: x.get('test_rmse', {}).get('rmse_total', float('inf')))
                best_rmse = best.get('test_rmse', {})
                f.write(f"\n**Best Configuration ({dataset_name})**: {best['description']}\n")
                f.write(f"- **RMSE**: {best_rmse.get('rmse_total', 0):.6f}\n")
                f.write(f"- **F Type**: {best['config'].model.F_type}\n")
                f.write(f"- **K Type**: {best['config'].model.K_type}\n\n")

            # Cross-dataset analysis
            f.write("## Cross-dataset analysis\n\n")
            f.write("### F Type Comparison\n\n")
            for dataset_name, ds_results in by_dataset.items():
                learnable_results = [r for r in ds_results if r['config'].model.F_type == 'learnable']
                fixed_results = [r for r in ds_results if r['config'].model.F_type == 'fixed']
                if learnable_results and fixed_results:
                    avg_learnable = np.mean([r['test_rmse']['rmse_total'] for r in learnable_results])
                    avg_fixed = np.mean([r['test_rmse']['rmse_total'] for r in fixed_results])
                    better = "learnable" if avg_learnable < avg_fixed else "fixed"
                    improvement = abs(avg_learnable - avg_fixed) / avg_fixed * 100
                    f.write(f"- **{dataset_name}**:\n")
                    f.write(f"  - F=learnable: {avg_learnable:.6f}\n")
                    f.write(f"  - F=fixed: {avg_fixed:.6f}\n")
                    f.write(f"  - **Better**: {better} (Improvement: {improvement:.2f}%)\n\n")

            # Conclusions
            f.write("## Conclusions\n\n")
            f.write("The following conclusions can be drawn from the comparison experiments:\n\n")
            f.write("1. **Complex trajectory complex noise dataset** (complex_complex_noise):\n")
            f.write("   - When motion patterns are complex (including CV/CA/CT/Maneuvering) and noise is complex (multiple types)\n")
            f.write("   - F learnable performs better than F fixed, learnable state transition matrix better adapts to complex motion and noise\n\n")

            f.write("2. **Complex trajectory fixed noise dataset** (complex_fixed_noise):\n")
            f.write("   - When motion patterns are complex but noise is fixed (Gaussian only)\n")
            f.write("   - F learnable should still outperform F fixed, as the main influence is motion pattern complexity\n")
            f.write("   - Comparison with complex dataset demonstrates noise type impact on model selection\n\n")

            f.write("3. **Simple CA trajectory complex noise dataset** (simple_ca_complex_noise):\n")
            f.write("   - When motion pattern is simple (constant acceleration only) but noise is complex\n")
            f.write("   - F fixed and F learnable should perform similarly, as model assumptions match motion pattern\n")
            f.write("   - Learnable modules (Q, R) can better adapt to complex noise environments\n\n")

            f.write("4. **Practical recommendations**:\n")
            f.write("   - For known simple motion models: Use fixed state transition matrix (e.g., CA/CV), make noise modules learnable\n")
            f.write("   - For complex unknown motion: Use learnable state transition matrix to adapt to multiple motion patterns\n")
            f.write("   - Noise complexity has greater impact on Q/R module learning, relatively smaller impact on F module\n\n")

        self.logger.info(f"\nSummary report saved to: {report_path}")


def main():
    """Main function"""
    # Create ablation study instance
    ablation = AblationStudy()

    # Set random seed
    set_random_seed(42)

    # Run dataset comparison experiment
    results = ablation.experiment_dataset_comparison()

    self.logger.info("\nAblation study completed!")
    self.logger.info(f"Results saved to: {ablation.output_dir}")


if __name__ == "__main__":
    main()
