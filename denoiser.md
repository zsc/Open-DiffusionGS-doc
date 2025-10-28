# denoiser.py 源码解析

`denoiser.py` 文件定义了 `DGSDenoiser` 类，这是 DiffusionGS 模型的核心，也是在 `PointDiffusionSystem` 中被引用的 `shape_model`。这个类继承自 `BaseModule`，其主要职责是在扩散过程的每个时间步，根据带噪声的图像和相机信息，预测出 3D 高斯分布（Gaussian Splatting）的参数。它是连接 2D 图像信息和 3D 场景表示的关键模块。

## 核心类与功能

### `DGSDenoiser` 类

`DGSDenoiser` 是一个基于 Transformer 的去噪模型，专门用于从 2D 图像（及其姿态信息）生成 3D 高斯分布。

#### `configure(self)`

*   **功能**: 初始化 `DGSDenoiser` 的所有子模块。这些模块协同工作，完成了从输入处理到最终高斯参数输出的整个流程。
*   **核心子模块**:
    *   `t_embedder`: `TimestepEmbedder` 的实例，负责将扩散过程中的标量时间步 `t` 转换为高维向量嵌入，以便作为条件注入到 Transformer 模型中。
    *   `image_tokenizer`: 一个 `nn.Sequential` 模块，负责将输入的图像“通证化”（Tokenize）。它首先将图像分割成多个小块（Patch），然后通过一个线性层将每个 Patch 转换为一个高维度的特征向量（Token）。
    *   `gaussians_pos_embedding`: 一个可学习的 `nn.Parameter`，它为“全局高斯”提供位置嵌入。这些全局高斯不与图像的任何特定部分对齐，而是用于捕捉物体的整体结构。
    *   `transformer`: 一个由多个 `DiTBlock`（Diffusion Transformer Block）组成的 `nn.ModuleList`。这是模型的主体，负责处理连接后的图像 Tokens 和全局高斯 Tokens，并根据时间步嵌入进行条件化处理。
    *   `upsampler`: `GaussiansUpsampler` 的实例，负责将 Transformer 输出的全局高斯 Tokens 解码为最终的 3D 高斯参数（xyz, features, scaling, rotation, opacity）。
    *   `image_token_decoder`: `ImageTokenDecoder` 的实例，负责将 Transformer 输出的图像对齐 Tokens 解码为对应像素的 3D 高斯参数。
    *   `gs_renderer`: `Renderer` 的实例，用于将生成的高斯分布渲染成 2D 图像。

#### `image_to_gaussians(self, images, ray_o, ray_d, t, training=False)`

*   **功能**: 这是 `DGSDenoiser` 的核心前向传播方法，实现了从带噪声图像到 3D 高斯分布的完整转换逻辑。
*   **流程**:
    1.  **输入准备**: 将输入的（带噪声的）图像与相机光线信息（`ray_o`, `ray_d`，即光心和方向）结合，形成“姿态化”的图像 `posed_images`。这些光线信息为模型提供了关键的 3D 几何线索。
    2.  **通证化 (Tokenization)**: 使用 `image_tokenizer` 将 `posed_images` 转换为图像 Tokens (`img_tokens`)。
    3.  **时间步嵌入**: 使用 `t_embedder` 将时间步 `t` 转换为向量嵌入。
    4.  **Transformer 处理**: 将“全局高斯”的 `gaussians_pos_embedding` 和 `img_tokens` 连接起来，送入 `transformer` 网络。Transformer 在时间步嵌入的条件下，对这些 Tokens 进行深度处理和信息交互。
    5.  **解码 (Decoding)**: 将 Transformer 的输出重新分离为全局高斯 Tokens 和图像对齐 Tokens。
        *   全局高斯 Tokens 通过 `upsampler` 解码，生成全局高斯的参数。
        *   图像对齐 Tokens 通过 `image_token_decoder` 解码，生成与图像像素对齐的高斯参数。
    6.  **XYZ 坐标校正**: 对图像对齐的高斯分布的 `xyz` 坐标进行一个关键的校正。这一步利用相机光线信息，确保这些高斯点被“投射”到 3D 空间中的正确位置，从而与输入图像的几何关系保持一致。
    7.  **输出**: 返回所有高斯（全局 + 图像对齐）的完整参数集合，包括 `xyz`, `features`, `scaling`, `rotation`, `opacity`。

#### `render_gaussians(self, ...)`

*   **功能**: 接收 `image_to_gaussians` 生成的高斯参数，并利用 `gs_renderer` 将这些 3D 高斯分布渲染成指定视角的 2D 图像。

## 关键设计理念

*   **两种高斯分布的结合**: 模型同时预测两种类型的高斯分布，这是一个核心设计。
    *   **全局高斯 (Global Gaussians)**: 数量较少，负责捕捉物体的整体、低频结构。
    *   **图像对齐高斯 (Image-Aligned Gaussians)**: 数量较多，与输入图像的像素对齐，负责捕捉物体的细节和高频信息。
*   **扩散 Transformer (DiT)**: 采用强大的 Transformer 架构作为去噪网络的主干，使其能够高效地处理图像和高斯 Tokens 序列，并根据时间步进行条件生成。
*   **光线索 (Plenoptic Cues)**: 将相机光线信息作为输入的一部分，为模型提供了必要的 3D 几何约束，使其能够更准确地推断 3D 结构。

## 总结

`denoiser.py` 中的 `DGSDenoiser` 是 DiffusionGS 能够从单张 2D 图像生成高质量 3D 模型的“魔法”所在。它通过一个精心设计的 Transformer 网络，巧妙地结合了全局和局部的 3D 表示，并利用光线信息作为几何引导，最终在扩散模型的框架下，实现了从噪声到精细 3D 高斯分布的生成。

```mermaid
graph TD
    A[输入: 带噪声图像, 相机光线, 时间步 t] --> B(DGSDenoiser);
    B --> C{image_to_gaussians 方法};
    C --> D[1. 准备输入 (图像 + 光线)];
    D --> E[2. 图像通证化];
    E --> F[3. 时间步嵌入];
    F --> G[4. 全局高斯 Token + 图像 Token];
    G --> H[5. Transformer 处理];
    H --> I[6. 解码 (全局 + 图像对齐)];
    I --> J[7. XYZ 坐标校正];
    J --> K[输出: 3D 高斯参数];
```
