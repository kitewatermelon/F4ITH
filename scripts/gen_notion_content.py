#!/usr/bin/env python
"""Emit Notion-flavored markdown for the downstream-agreement page."""
import json

d = json.load(open("/tmp/claude-1001/-home-yspark-RCT/905fc9ef-98c7-4124-8b8a-b8d26ff0a52c/scratchpad/agreement_all.json"))
MODELS = ["spatial", "pooled", "RCDM", "FeatInv"]
DSETS = ["MIMIC", "NIH", "CheXpert"]


def mark(val, best, second, hib, fmt="{:.3f}"):
    s = fmt.format(val)
    eps = 1e-9
    if abs(val - best) < eps:
        return f"**{s}**"
    if abs(val - second) < eps:
        return f'<span underline="true">{s}</span>'
    return s


def rank_vals(vals, hib):
    order = sorted(vals, reverse=hib)
    return order[0], (order[1] if len(order) > 1 else order[0])


def row(cells):
    return "<tr>\n" + "\n".join(f"<td>{c}</td>" for c in cells) + "\n</tr>"


def table(header, rows):
    return ('<table header-row="true">\n' + row(header) + "\n" +
            "\n".join(rows) + "\n</table>")


out = []

# ---- Table 1: segmentation macro ----
rows = []
for ds in DSETS:
    dice = [d[ds][m]["seg"]["macro"]["dice"] for m in MODELS]
    iou = [d[ds][m]["seg"]["macro"]["iou"] for m in MODELS]
    bd, sd = rank_vals(dice, True)
    bi, si = rank_vals(iou, True)
    n = d[ds]["spatial"]["seg"]["macro"]["n"]
    for i, m in enumerate(MODELS):
        rows.append(row([ds if i == 0 else "", m,
                         mark(dice[i], bd, sd, True),
                         mark(iou[i], bi, si, True),
                         str(n) if i == 0 else ""]))
t1 = table(["dataset", "model", "Dice ↑", "IoU ↑", "N pairs"], rows)

# ---- Table 2: classification 7-model mean ----
rows = []
for ds in DSETS:
    pr = [d[ds][m]["cls"]["mean_pearson"] for m in MODELS]
    ma = [d[ds][m]["cls"]["mean_mae"] for m in MODELS]
    ba = [d[ds][m]["cls"]["mean_bin_agree"] for m in MODELS]
    bpr, spr = rank_vals(pr, True)
    bma, sma = rank_vals(ma, False)
    bba, sba = rank_vals(ba, True)
    for i, m in enumerate(MODELS):
        rows.append(row([ds if i == 0 else "", m,
                         mark(pr[i], bpr, spr, True),
                         mark(ma[i], bma, sma, False, "{:.4f}"),
                         mark(ba[i], bba, sba, True)]))
t2 = table(["dataset", "model", "Pearson ↑", "MAE ↓", "BinAgree ↑"], rows)

# ---- Table 3: per-organ Dice grouped by anatomy, per dataset ----
GROUPS = {
    "대형 실질·실루엣": ["Left Lung", "Right Lung", "Facies Diaphragmatica", "Heart", "Mediastinum"],
    "골격": ["Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Spine"],
    "미세·저대비 종격동": ["Aorta", "Weasand", "Left Hilus Pulmonis", "Right Hilus Pulmonis"],
}


def organ_table(ds):
    rows = []
    for g, orgs in GROUPS.items():
        for j, o in enumerate(orgs):
            vals = [d[ds][m]["seg"]["organs"][o]["dice"] for m in MODELS]
            b, s = rank_vals(vals, True)
            rows.append(row([g if j == 0 else "", o] + [mark(v, b, s, True) for v in vals]))
        # group-mean row (not marked)
        gm = [sum(d[ds][m]["seg"]["organs"][o]["dice"] for o in orgs) / len(orgs) for m in MODELS]
        rows.append(row(["", "▸ 그룹 평균"] + [f"*{v:.3f}*" for v in gm]))
    return table(["group", f"organ ({ds})"] + MODELS, rows)


t3_mimic = organ_table("MIMIC")
t3_nih = organ_table("NIH")
t3_chex = organ_table("CheXpert")

# ---- Table 4: per-classifier Pearson (all datasets) ----
clfs = list(d["MIMIC"]["spatial"]["cls"]["per_model"].keys())
short = {c: c.replace("densenet121-res224-", "") for c in clfs}
rows = []
for ds in DSETS:
    for j, c in enumerate(clfs):
        vals = [d[ds][m]["cls"]["per_model"][c]["pearson"] for m in MODELS]
        b, s = rank_vals(vals, True)
        rows.append(row([ds if j == 0 else "", short[c]] +
                        [mark(v, b, s, True) for v in vals]))
t4 = table(["dataset", "classifier"] + MODELS, rows)

intro = """표기: **굵게 = 각 데이터셋 내 열 최고**, <span underline="true">밑줄 = 2위</span>. 모든 이미지를 **224로 resize**한 뒤 측정(spatial/pooled=native 224, RCDM 128→224, FeatInv 256→224). pair = 같은 conditioning real 이미지 ↔ 그 재구성(gen). 값이 높을수록(seg Dice/IoU·cls Pearson·BinAgree) 재구성이 **원본의 해부·병리 신호를 더 충실히 보존**."""

signif = """## 6.0 실험의 의의 — 왜 downstream agreement인가

기존 표 3·4는 **분포 사실성(FID/FRD)** 과 **표현공간 보존(CheXWorld cosine/ratio/rank)** 을 측정했다. 그러나 두 축 모두 답하지 못하는 질문이 있다: *"재구성 이미지가 자기 conditioning 원본의 **해부 구조와 병리 소견**을 픽셀·판독 수준에서 실제로 보존하는가?"*

- **FID는 집합-대-집합 통계**다. gen 집합이 real 집합과 "비슷한 분포"이기만 하면 낮아지며, **개별 pair가 서로 대응하는지는 보지 않는다.** 원본 A의 심장 위치를 B의 것으로 바꿔 생성해도 분포만 맞으면 FID는 좋을 수 있다.
- **CheXWorld 표현거리**는 조건 인코더 **자기 자신**의 임베딩공간에서 재는 지표라, 그 인코더가 놓치는 축(미세 골격·저대비 종격동 구조)은 반영되지 않고, "인코더 관점의 보존"에 갇힌다(self-referential).

이 실험은 **완전히 독립적인 제3의 전문가 모델**(ChestX-Det PSPNet 해부 분할기, 7개 데이터셋 출처의 DenseNet 병리 분류기)을 판정자로 세워, 재구성이 **임상적으로 의미 있는 정보**(장기의 위치·형태, 병리 예측)를 보존하는지 pair 단위로 검증한다. 즉 *"이 재구성을 실제 판독 파이프라인에 넣었을 때 원본과 같은 결론이 나오는가"* 를 근사한다. RCT의 목표가 **표현 → 이미지 역변환(inversion)의 충실도**임을 감안하면, 이는 표현 지표보다 한 단계 더 실체적인(grounded) 증거다.

특히 **분할기·분류기 모두 CheXWorld와 무관하게 학습된 외부 모델**이므로, 여기서 spatial이 이기면 그 우위는 "조건 인코더에 유리하게 최적화된 착시"가 아니라 **인코더-독립적인 재구성 품질**임을 보장한다."""

metric_detail = """## 6.1 Metric의 의미

**공통 설계 — pairing & 224 통제.** 모든 지표는 `gen[i]`와 `real[i]`(= i번째 conditioning 원본)를 **한 쌍**으로 비교한다(집합 통계가 아니라 pair 단위). 모델별 native 해상도가 달라(spatial/pooled 224, RCDM 128, FeatInv 256) 판정자 입력을 **전부 224로 통일**했다 — 해상도가 아니라 재구성 내용 자체를 비교하기 위함(224 resize 재실험에서 FeatInv 256 vs 224 차이가 macro Dice ±0.01로 무시할 수준임을 확인).

**① Segmentation — PSPNet Dice / IoU (해부 구조 재현).**
- ChestX-Det **PSPNet**(14 해부구조, native 512에서 판정)을 gen·real에 각각 적용해 장기별 이진 마스크(sigmoid>0.5)를 얻고 **같은 장기 마스크가 얼마나 겹치는지**를 잰다.
- **Dice** = 2|A∩B| / (|A|+|B|) — 겹침의 조화평균적 척도, 0(불일치)~1(완전일치). **IoU** = |A∩B| / |A∪B| — 같은 겹침을 더 엄격히(교집합/합집합) 본다. 둘은 단조 관계라 방향은 같고, IoU가 항상 더 낮다.
- 해석: Dice가 높다 = 재구성이 원본과 **같은 위치·같은 형태**로 그 장기를 그렸다 = 해부학적으로 충실. 낮다 = 장기가 이동·왜곡·소실.
- **둘 다 빈 마스크(both-empty)는 그 장기 평균에서 제외.** 원본에 애초에 안 보이는 구조(예: 일부 영상의 쇄골)를 "완벽 일치(1.0)"로 세어 점수를 부풀리는 것을 막는다 → N(valid pair)이 장기마다 다른 이유.

**② Classification — DenseNet 예측 추종 (병리 소견 재현).**
- xrv **DenseNet-121 7종**(학습 출처: all·rsna·nih·pc·chex·mimic_nb·mimic_ch)으로 18개 흉부 병리 확률을 gen·real에 각각 예측하고, **gen 예측이 real 예측을 얼마나 따라가는지**를 잰다. 분류기를 7개나 쓰는 이유: 특정 데이터셋 편향을 배제하고 **판정자에 무관하게** 강건한지 보기 위함.
- **Pearson r** = 데이터셋 전체에 걸쳐 gen 확률이 real 확률과 **함께 오르내리는 정도**(추세 보존, ↑). 핵심 지표 — 병리 신호의 상대적 순위가 보존되는지.
- **MAE** = |p_gen − p_real| 평균(↓). 확률의 **절대적 근접도**.
- **BinAgree@0.5** = 0.5 임계로 이진화한 예측 라벨의 일치율(↑). "양성/음성 판정"이 뒤집히지 않는 비율 — 가장 임상 결정에 가까운 척도.
- **informative 필터(real std≥0.02):** 해당 데이터셋에서 real 예측이 거의 상수인(변별력 없는) 병리는 상관계수가 불안정하므로 평균에서 제외. 그래서 분류기마다 유효 병리 수가 다르다(rsna는 2개뿐 → 참고용).

**한계.** (a) **RCDM은 native 128을 224로 업샘플**해 판정 → 절대치가 다소 불리(정보량 상한). (b) 모든 재구성의 **conditioning이 224에서 인코딩**되므로, 재현 가능한 해부 디테일의 상한도 224 수준. (c) pairing은 파일명 정렬이 곧 동일 원본이라는 전제에 의존(카운트·정렬 일치 확인함). (d) 판정 모델 자체의 오류(특히 저대비 hilus)는 gen·real 양쪽에 동일하게 작용하므로 **상대 비교**에는 영향이 적으나, 절대 Dice의 천장을 낮춘다."""

concl = """**핵심 결론**
- **spatial(RCT token, patch cross-attn)이 세그·분류 전 지표·전 데이터셋에서 압도적 1위.** 재구성이 원본의 해부 구조(Dice 0.69–0.72)와 병리 예측(Pearson 0.91–0.93)을 가장 충실히 보존. 이는 METRIC 표4의 표현 보존(ratio·rank/MRR) 우위가 **다운스트림 의학적 의미(장기 형태·병리 판독)까지 이어짐**을 입증.
- **순위: spatial ≫ pooled ≳ RCDM ≈ FeatInv.** pooled(mean-pool 조건)은 spatial보다 크게 낮아 dense spatial 조건이 다운스트림 충실도의 핵심임을 재확인(표4 결론과 일치).
- **분류 격차는 out-of-distribution 분류기에서 가장 큼**: 어떤 데이터셋이든 chex 분류기는 네 모델 다 높지만(재구성이 CheXWorld=chexpert 계열로 학습돼 편향), nih·pc·rsna 분류기에선 spatial만 0.9대 유지하고 pooled/RCDM/FeatInv는 0.5–0.7로 붕괴 — spatial이 분류기 종류에 관계없이 강건.
- **세그는 큰 구조(폐·횡격막)에선 모델 간 차이가 작지만**, 미세·저대비 구조(쇄골·대동맥·폐문 hilus·식도 Weasand)에서 spatial의 우위가 극대화(표3). 종격동/골격 미세구조 재현이 모델을 가르는 지점.
- ⚠️ **RCDM은 native 128을 224로 업샘플**해 측정한 값이라 seg/cls 절대치가 다소 불리할 수 있음(정보량 한계). 그럼에도 pooled과 비슷하거나 낮음.

*측정 스크립트: `scripts/eval_seg_agreement.py`, `scripts/eval_cls_agreement.py`, `scripts/run_all_metrics.sh`(전 조합 224 강제), `scripts/aggregate_agreement.py`. 원자료 CSV: 각 `<model>/<dataset>/seg_agreement_r224/` · `cls_agreement_r224/`.*"""

seg_organ_intro = """## 6c. 장기별 Segmentation Dice — 해부 그룹별 정리

14개 해부구조를 **재현 난이도에 따라 3그룹**으로 묶었다. 각 표는 열=4모델·행=장기(그룹별 정렬), 값=pair Dice(↑), 각 행 **굵게=최고/밑줄=2위**, 각 그룹 끝의 *그룹 평균*은 기울임. 핵심 패턴: **구조가 작고 저대비일수록 모델 간 격차가 벌어지고 spatial의 우위가 커진다.**

- **대형 실질·실루엣**(폐·횡격막·심장·종격동): 네 모델 모두 0.76~0.89로 접전 — 전역 흉부 윤곽은 누구나 잡는다.
- **골격**(쇄골·견갑골·척추): spatial이 2위와 0.13~0.15 Dice 앞섬. 얇은 뼈 경계 재현에서 갈린다.
- **미세·저대비 종격동**(대동맥·식도Weasand·좌우 폐문): 절대 Dice 자체가 낮은(0.2~0.5) 최난도 영역. spatial이 pooled 대비 +0.10, RCDM/FeatInv 대비 +0.19~0.29로 **가장 크게 벌어진다.** 재구성 모델을 실제로 가르는 지점."""

content = "\n".join([
    intro,
    "# 6. Downstream 해부·병리 재현도 (Segmentation / Classification agreement)",
    signif,
    metric_detail,
    "## 6a. Segmentation 일치 (macro) — PSPNet 14-organ Dice/IoU",
    "**표 6a. 해부구조 재현도(요약).** gen·real을 PSPNet으로 분할 후 pair Dice/IoU(macro=14장기 평균, ↑ 높을수록 원본 해부 보존). N=데이터셋별 pair 수. 장기별 상세는 6c.",
    t1,
    "## 6b. Classification 일치 (macro) — DenseNet 7-classifier 예측 추종",
    "**표 6b. 병리 예측 추종도(요약).** 7개 DenseNet 분류기 평균. Pearson=추세 보존(↑), MAE=확률 절대차(↓), BinAgree=@0.5 이진 판정 일치율(↑). 분류기별 상세는 6d.",
    t2,
    seg_organ_intro,
    "**표 6c-1. MIMIC 장기별 Dice** (n=3,403).",
    t3_mimic,
    "**표 6c-2. NIH 장기별 Dice** (n=25,596).",
    t3_nih,
    "**표 6c-3. CheXpert 장기별 Dice** (n=202).",
    t3_chex,
    "## 6d. 분류기별 Pearson (3 datasets) — spatial의 분류기 강건성",
    "**표 6d. 분류기별 Pearson.** 7개 DenseNet(데이터셋 학습 출처) × 3 held-out set. chex 분류기는 전 모델 높지만(재구성이 chexpert 계열로 학습돼 편향), nih·pc·rsna에선 spatial만 0.9대 유지.",
    t4,
    concl,
])

open("/tmp/claude-1001/-home-yspark-RCT/905fc9ef-98c7-4124-8b8a-b8d26ff0a52c/scratchpad/notion_page.md", "w").write(content)
print("wrote notion_page.md", len(content), "chars")
