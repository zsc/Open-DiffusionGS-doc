# launch.py 源码解析

`launch.py` 是一个通用的启动脚本，用于执行基于 PyTorch Lightning 的训练、验证、测试和导出任务。它本身不包含特定于某个模型（如物体生成或场景重建）的逻辑，而是通过解析命令行参数和配置文件来动态加载和运行指定的模块。

## 核心功能

`launch.py` 的核心功能是设置 PyTorch Lightning 的训练环境，并根据用户的指令启动相应的流程。

### 1. 参数解析

脚本使用 `argparse` 来解析命令行参数，其中最重要的几个参数是：

*   `--config`: 指向一个 YAML 配置文件，这是**最核心的参数**，因为它定义了整个实验的所有配置，包括使用哪个数据模块、哪个系统模块、模型的超参数、优化器设置等。
*   `--gpu`: 指定要使用的 GPU 设备。
*   `--train`, `--validate`, `--test`, `--export`: 这些是互斥的标志，用于指定要执行的操作模式（训练、验证、测试或导出）。

### 2. 动态模块加载

`launch.py` 的设计精髓在于其**动态性**。它不硬编码任何特定的数据或模型，而是使用 `diffusionGS.find` 函数，根据配置文件中的字符串名称来查找并实例化对应的类。

*   **数据模块 (DataModule)**: `dm = diffusionGS.find(cfg.data_type)(cfg.data)`
    *   它读取配置文件中的 `data_type` 字段（例如 `"Re10k-datamodule"`），然后在注册表中找到对应的类，并用 `cfg.data` 中的配置来实例化它。
*   **系统模块 (System)**: `system: BaseSystem = diffusionGS.find(cfg.system_type)(cfg.system, ...)`
    *   同理，它读取 `system_type` 字段（例如 `"diffusion-gs-scene-system"`），找到对应的 `pl.LightningModule` 子类（如 `PointDiffusionSystem`），并用 `cfg.system` 中的配置来实例化它。

### 3. PyTorch Lightning `Trainer` 的配置和执行

脚本的核心是配置和使用 `pytorch_lightning.Trainer`。

*   **回调 (Callbacks)**: 它会根据配置设置一系列的回调，例如：
    *   `ModelCheckpoint`: 用于保存模型的检查点。
    *   `LearningRateMonitor`: 监控学习率的变化。
    *   `EMA` (Exponential Moving Average): 如果启用，会使用 EMA 来平滑模型权重。
    *   以及一些自定义的回调，如 `CodeSnapshotCallback`（保存代码快照）和 `ConfigSnapshotCallback`（保存配置快照）。
*   **日志记录器 (Loggers)**: 设置 `TensorBoardLogger` 和 `CSVLogger` 来记录训练过程中的指标。
*   **启动训练/评估**: 最后，根据命令行参数，调用 `Trainer` 的相应方法来启动任务：
    *   `trainer.fit(system, datamodule=dm, ...)`: 启动训练。
    *   `trainer.validate(system, datamodule=dm, ...)`: 启动验证。
    *   `trainer.test(system, datamodule=dm, ...)`: 启动测试。
    *   `trainer.predict(system, datamodule=dm, ...)`: 启动预测（用于导出）。

## 总结

`launch.py` 是整个项目的“点火钥匙”或“总开关”。它将数据、模型和训练过程解耦，使得研究人员可以仅仅通过修改配置文件，就能灵活地组合不同的数据和模型，进行各种实验，而无需修改代码。

**执行流程图**:

```mermaid
graph TD
    A[命令行: launch.py --config config.yaml --train] --> B{main 函数};
    B --> C[加载配置文件 config.yaml];
    C --> D[设置 GPU 和随机种子];
    D --> E{动态加载模块};
    E --> F[实例化 DataModule (由 cfg.data_type 决定)];
    E --> G[实例化 System (由 cfg.system_type 决定)];
    G --> H[配置 Trainer (Callbacks, Loggers)];
    H --> I{根据 --train/--validate 等参数};
    I -- train --> J[调用 trainer.fit];
    I -- validate --> K[调用 trainer.validate];
    I -- test --> L[调用 trainer.test];
```

理解 `launch.py` 对于掌握如何在本项目中运行训练和评估至关重要。它清晰地展示了如何利用 PyTorch Lightning 和配置文件驱动的方式来组织一个灵活的深度学习项目。
