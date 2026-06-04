# Research Problem: A Heuristic Placement Framework on 3D Design
B11901047 郭祐嘉

Reference: [2023  CAD Contest Problem B - 3D Placement with Macros](https://drive.google.com/file/d/1PJOSECe0sCDGzJoQrQWGIzTnIyUVOr65/view)

#### Problem Overview

由於我未來打算往 EDA 領域發展，並且研究主題會與 Physical Design 相關，所以我想要利用這個 Final Project 的機會對於 CVXP, QP 在此領域的應用有更深入的研究。
本研究旨在開發一個高效能的 **3D Placer**，以解決 ICCAD 2023 Problem B 所定義的挑戰。該問題的核心難點在於需要同時處理三大耦合的子問題：
**電路分割 (Partitioning)**：如何智慧地將百萬級的元件分配到兩個不同的晶片上。
**混合尺寸佈局 (Mixed-Size Placement)**：如何在充滿大型 Macro 和大量 Standard Cell 的擁擠空間中，找到無重疊且線長最佳的佈局。
**3D 整合 (3D Integration)**：如何決定跨晶片連接點 (Terminal) 的位置，以平衡線長縮短與額外成本之間的取捨。

我預計將使用 Quadratic Programming and Heuristic method 來完成這一題。

### Part I：大方向主軸

#### 方向 A：Robust Mixed-Size Analytical Placement Engine
**建立並求解 QP 模型**來達成全局佈局。如何處理 **Macros 的存在**以及**Non-Overlap** 這兩個 issue。目標是開發一個能產生高品質、低重疊度的全局佈局，並為後續的  Legalization 步驟打下良好基礎的 Placer。

#### 方向 B：Cost-Aware 3D Terminal Optimization
這個方向專注於 3D 佈局的獨特挑戰：**Hybrid Bonding Terminals**。每個 Terminal 雖然能縮短潛在的跨晶片線長，但本身也帶有成本。此方向旨在研究如何建立模型來量化這個「利弊」，並開發演算法來自動決定**哪些 Net 值得跨晶片連接**，以及**將 Terminal 放置在何處**才能取得最大收益。

### Part II：Detailed Research Topics

#### 方向 A：穩健的混合尺寸解析式佈局引擎

**A.1. 核心二次規劃 (QP) 模型建立：**
研究如何將 Standard Cells 之間的連接關係，轉化為一個 **Sum of Squared Wirelength** 的二次目標函數 `(xᵀQx + yᵀQy)`。使用高效的數值方法 (如 Conjugate Gradient) 來求解這個大型稀疏線性方程組。

**A.2. 處理重疊 (Non-Overlap) 的策略研究：**
**策略一 (密度懲罰法)**：在 QP 目標函數中，加入一項「密度」懲罰項，當某個區域的 cell 密度過高時，施加一個「排斥力」將其推開。
**策略二 (後處理合法化)**：在 QP 解完得到一個重疊的全局佈局後，開發一個獨立的 Legalizer，用最小的擾動將所有 cell 推到合法的、不重疊的 row-based 位置上。

**A.3. Macro 佈局策略**：
**策略一 (迭代式固定與求解)**：在主迴圈中，先固定 Macro 位置，用 QP 解 Standard Cell 位置；然後固定 Standard Cell 的重心，用 **Simulated Annealing** 或其他 heuristic 方法微調 Macro 的位置和方向。
**策略二 (soft module)**：將 Macro 視為由多個小點組成的 soft module，在 QP 中加入約束使其保持形狀。

#### 方向 B：成本敏感的 3D 連接點優化

**B.1. Terminal 位置的解析式解法**：
對於一個已經確定要跨晶片的 Net，其 Terminal 的最佳位置在哪裡？可以證明，最佳位置是該 Net 所有 Pin 在 x 和 y 方向上的**中位數**。可以研究如何將 Terminal 視為一個「可移動的 Pin」並整合進 QP 模型中。

**B.2. 跨晶片決策的動態優化**：
在迭代過程中，動態評估某個 Net：如果它保持在同一個 Die 內，線長是多少？如果它被切開（並增加一個 Terminal 成本），預估的線長又是多少？根據這個 `(線長收益 vs. Terminal 成本)` 的比較，來動態決定是否要將它移到另一個 Die。