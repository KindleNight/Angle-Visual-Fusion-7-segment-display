# 基于视觉和角度的晶体管识别

### 项目致谢
本项目是一个二次开发的项目，是基于著名的开源目标检测框架Ultralytics（ https://github.com/ultralytics/ultralytics ）这个开源项目改编而成。在此特别感谢原作者团队的卓越工作，为本项目提供了强大的底层算法支持。

### 项目简介
本项目专注于特定的场景（如：七段、八段晶体管识别等）。在目标由明显的角度特征的情况下，均适合使用该项目。针对这一类型场景，我们对原生的 YOLO 架构进行了深度的修改和优化。

### 自定义架构
* 在 **Backbone** 部分引入了自定义的 `Angle` 模块，用于提取和融合角度信息。
* 构建了独特的双 **Head** 结构（左侧 AP 系列 Head 与右侧 P 系列 Head）。
* 在 Neck 部分引入了 `Data_fusion` 模块，实现了多尺度特征的高效融合，最终汇总至 Detect 头。

以下是本项目自定义网络架构的详细图解：
 
<img width="600" height="1000" alt="架构图" src="https://github.com/user-attachments/assets/4ff5e0b4-3403-411f-a65c-fa5344ace676" />
