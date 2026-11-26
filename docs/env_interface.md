# WeedEnv arayüzü

`sim/weed_env.py` içindeki `WeedEnv`, PettingZoo `ParallelEnv` arayüzünü uygular. Drone'lar otları algılayıp yer robotlarına mesajla bildirir; robotlar bu raporlarla hareket eder ve ilaçlar.

```python
from sim.weed_env import WeedEnv, WeedEnvConfig

env = WeedEnv(WeedEnvConfig(grid_path="data/grids_64/parcel_000_first000_gt_64x64.npy"))
observations, infos = env.reset(seed=0)
```

## Ajanlar

- Adlar: `drone_0 … drone_{N-1}`, ardından `robot_0 … robot_{M-1}`. N ve M 1–4 arasında; varsayılan 2+2.
- `possible_agents` ve `agents` her zaman bu sabit sırayla gelir. `action_masks()`, `state()` ve EPyMARL yardımcıları da aynı sırayı kullanır.
- Ajanlar aynı hücrede bulunabilir; çarpışma modeli yok.
- Başlangıç konumları tarla içinde tekdüze rastgeledir ve tohuma bağlıdır.

## Hücre kodları

`detector/convert_to_grid64.py` ile aynı: `0` toprak, `1` ürün, `2` ot.

## Eylemler

Tüm ajanlar aynı `Discrete(6)` uzayını kullanır.

| ID | Eylem | Drone | Robot |
|---|---|---|---|
| 0 | STAY | ✔ | ✔ |
| 1 | UP (satır − 1) | ✔ | ✔ |
| 2 | DOWN (satır + 1) | ✔ | ✔ |
| 3 | LEFT (sütun − 1) | ✔ | ✔ |
| 4 | RIGHT (sütun + 1) | ✔ | ✔ |
| 5 | SPRAY | ✘ | ✔ |

- Izgara dışına çıkan hareket STAY sayılır ve hareket maliyeti düşmez.
- Maskeyle kapalı bir eylem gönderilirse STAY uygulanır.

### İlaçlama (SPRAY)

- Yalnızca robotlar ilaçlayabilir. Robot hareket etmeden bulunduğu hücreyi ilaçlar ve `spray_cost` kadar batarya harcar.
- Hücre sayımı:
  - Gerçek ot hücresi "ilaçlanan ot" (W) sayılır.
  - Ot olmayan hücre, ürün ya da toprak fark etmeksizin "gereksiz ilaçlama" (U) sayılır.
- Daha önce ilaçlanmış bir hücre tekrar ilaçlanırsa ne W ne U sayılır, ama batarya yine harcanır. Aynı adımda iki robot aynı hücreyi ilaçlarsa da hücre tek sayılır.
- W ve U, o adımın ödülü için doğrudan `marl/reward.py`'ye verilir.
- İlaçlama, aynı adımdaki hareketlerden önce, ajanların adım başındaki konumlarında uygulanır.

## Eylem maskesi

`infos[agent]["action_mask"]`: uzunluğu 6 olan `int8` 0/1 dizi. `reset` ve her `step` sonrasında verilir. `env.action_masks()` tüm ajanlar için `(N+M, 6)` dizi döndürür.

| Durum | Maske |
|---|---|
| Drone, batarya ≥ hareket maliyeti | `[1, 1, 1, 1, 1, 0]` |
| Robot, batarya ≥ hareket ve ilaçlama maliyeti | `[1, 1, 1, 1, 1, 1]` |
| Batarya < hareket maliyeti | Hareketler 0 |
| Robot, batarya < ilaçlama maliyeti | SPRAY 0 |
| Batarya 0 | `[1, 0, 0, 0, 0, 0]` |

Izgara kenarındaki hareketler maskelenmez; bunlar STAY gibi davranır.

## Batarya modeli

- Her ajanın bir sayacı vardır. Başlangıç değeri `drone_battery` ya da `robot_battery`.
- Gerçekten yapılan hareket `drone_move_cost` ya da `robot_move_cost` kadar düşürür.
- STAY ve kenara çarpan hareket `stay_cost` kadar düşürür (varsayılan 0).
- SPRAY `spray_cost` kadar düşürür.
- Batarya 0'ın altına inmez.
- Bataryası biten ajan yalnızca STAY yapabilir. Drone'un algılaması ve mesaj göndermesi bataryadan bağımsız sürer.

## Algılama ve bilgi akışı

1. Her adımda, hareketten sonra her drone kendi etrafındaki `(2r+1)×(2r+1)` kareyi algılar (r = `detection_radius`).
2. Etiketleri `detector.detect(true_labels, rng)` üretir. Varsayılan `PerfectDetector` gerçek etiketi güven 1.0 ile döndürür. Sema'nın dedektörü `WeedEnv(config, detector=...)` ile takılır; arayüzü `sim/detection.py` içindedir.
3. Drone sonucu kendi keşfedilen haritasına yazar (`drone_explored`, `drone_weed`, `drone_confidence`).
4. Drone her robota bir mesaj gönderir; yani her adımda her bağlantıda bir mesaj gider. Mesajın içeriği `WeedReport`'tur: taranan blok sınırları, ot olarak algılanan hücreler ve güven değerleri.
5. Mesajlar `channel.transmit` üzerinden geçer. Ulaşan raporla robot taranan bloğu "keşfedilmiş" işaretler; bloktaki ot bilgisini rapordaki son haliyle günceller.
6. Robotlar ot konumlarını hiçbir zaman doğrudan görmez. Bildikleri her şey ulaşan mesajlardan gelir. Kaybolan mesajın içeriği robota hiç ulaşmaz.
7. Her ajan, kendi `k×k` penceresi içindeki hücrelerin ilaçlanmış olup olmadığını fiziksel olarak görür (`seen_sprayed`). Bu bilgi bir hücrenin ot olup olmadığını açığa çıkarmaz.

## Gözlem vektörü

`Box(-1, 1, (6 + 4·k² + 3·K,), float32)`. Burada k = `obs_window` (tek sayı) ve K = `n_reports`. Varsayılan değerlerle (k = 5, K = 5) uzunluk **121**.

| Aralık | Uzunluk | Alan | Değer |
|---|---|---|---|
| 0 | 1 | satır / (H − 1) | [0, 1] |
| 1 | 1 | sütun / (W − 1) | [0, 1] |
| 2 | 1 | batarya / kapasite | [0, 1] |
| 3–4 | 2 | tip one-hot: [drone, robot] | {0, 1} |
| 5 | 1 | Robot: son başarılı mesajdan bu yana geçen adım / max_steps (üst sınır 1). Drone: 0 | [0, 1] |
| 6 … 6+4k²−1 | 4·k² | Yerel pencere, kanal sırasıyla `[4, k, k]` düzleştirilmiş (aşağıda) | {0, 1} |
| son 3·K | 3·K | Robot: bilinen en yakın K ot. Her biri `[Δsatır/(H−1), Δsütun/(W−1), geçerli]`. Drone: 0 | [−1, 1] |

Yerel pencerenin kanalları (ajan merkezde, ızgara dışı hücreler 0):

| Kanal | Anlamı |
|---|---|
| 0 bilinen ot | Ajanın bildiği ve ilaçlandığını görmediği ot hücreleri. Drone için kendi algıladıkları, robot için ulaşan raporlardaki hücreler |
| 1 ilaçlanmış | Ajanın penceresinde gördüğü ilaçlanmış hücreler |
| 2 keşfedilmemiş | Drone için kendisinin taramadığı, robot için hiçbir raporun kapsamadığı hücreler |
| 3 diğer ajanlar | O hücrede kendisi dışında en az bir ajan varsa 1 |

En yakın K ot, Manhattan uzaklığına göre seçilir. Eşitlik olursa satır, sonra sütun önceliklidir. K'den az ot biliniyorsa kalan yuvalar 0'dır ve geçerlilik alanı da 0 olur.

## İletişim kanalı

- Config'deki `channel` alanı `perfect`, `bernoulli` ya da `gilbert_elliott` olabilir. Parametreler `channel_params` içinde verilir; kanal `sim.comms.make_channel` ile kurulur.
- Ortak arayüz (`sim/comms/base.py`):
  - `Channel.reset(rng)`: kanalı sıfırlar.
  - `Channel.transmit(messages) -> delivered`: mesajları gönderir, ulaşanları döndürür.
  - Sayaçlar: `sent` (gönderilen) ve `lost` (kaybolan).
- Her (gönderen, alıcı) bağlantısının durumu ayrı tutulur.

| Kanal | `channel_params` | Davranış |
|---|---|---|
| `perfect` | — | Hiçbir mesaj kaybolmaz |
| `bernoulli` | `{"p": 0.25}` | Her mesaj bağımsız olarak p olasılıkla kaybolur. Deney değerleri: p ∈ {0, 0.1, 0.25, 0.5} |
| `gilbert_elliott` | `{"mean_loss": 0.25, "mean_burst_length": 5, "e_g": 0.0, "e_b": 1.0}` ya da ham `{"p_gb", "p_bg", "e_g", "e_b"}` | Bağlantı başına Good/Bad Markov durumu. Deney değerleri: ortalama kayıp 0.25 ve 0.5 |

Gilbert-Elliott için `from_mean_loss` dönüşümü:
- p_BG = 1 / mean_burst_length
- π_B = (mean_loss − e_G) / (e_B − e_G)
- p_GB = π_B · p_BG / (1 − π_B)

e_G, e_B ve patlama uzunluğunun varsayılan değerleri (0, 1, 5) geçicidir; Nayak et al. (2020)'ye göre ayarlanacak.

Mesajın önce o anki duruma göre kaybolup kaybolmadığına karar verilir, ardından durum geçişi yapılır. Bağlantının ilk durumu durağan dağılımdan çekilir.

## Global state (`env.state()`)

Gözlemlerden ayrıdır ve yalnızca merkezi kritik içindir; hiçbir ajan gözlemine eklenmez. `Box(0, 1, (3·(N+M) + 3·S² + 1,), float32)`. Burada S = min(`state_map_size`, H, W). Varsayılan 2+2 ajan ve S = 16 ile uzunluk **781**.

| Bölüm | Uzunluk | İçerik |
|---|---|---|
| Ajanlar | 3·(N+M) | Ajan sırasıyla `[satır/(H−1), sütun/(W−1), batarya/kapasite]` |
| Kalan ot haritası | S² | Ot olup henüz ilaçlanmamış hücre oranı, S×S blok ortalaması |
| İlaçlanmış haritası | S² | İlaçlanmış hücre oranı, S×S blok ortalaması |
| Keşif haritası | S² | Herhangi bir drone'un taradığı hücre oranı, S×S blok ortalaması |
| İlerleme | 1 | adım / max_steps |

## Ödül

- `marl/reward.py` içindeki `calculate_reward(W, U, config.reward)` kullanılır: `10·W − 5·U − 0.1`.
- Takım ödülü her adımda **bir kez** hesaplanır ve her ajana aynı değer verilir. EPyMARL'da toplanırken yalnızca bir ajanınki alınmalıdır.
- W bu adımda ilaçlanan ot sayısı, U gereksiz ilaçlama sayısıdır; ilaçlama olmayan adımda ödül −0.1.

## Bitiş koşulları

| Bayrak | Koşul |
|---|---|
| `terminations` | Tüm gerçek ot hücreleri ilaçlandı **veya** tüm robotların bataryası, STAY dışında hiçbir eylemi yapamayacak kadar azaldı |
| `truncations` | `step_count ≥ max_steps` |

- İki bayrak birbirinden bağımsız hesaplanır; aynı adımda ikisi de True olabilir.
- Bölüm bitince `env.agents` boşalır.

## Info anahtarları

| Anahtar | Ne zaman | İçerik |
|---|---|---|
| `action_mask` | reset ve her step | 6 uzunluklu `int8` maske |
| `team_reward` | her step | O adımın takım ödülü |
| `episode` | bölümün son adımı | Aşağıdaki özet sözlüğü (tüm ajanlarda aynı) |

`episode` özeti:
- `steps`: bölümün kaç adım sürdüğü.
- `terminated`, `truncated`: bitiş bayrakları.
- `end_reasons`: `all_weeds_sprayed`, `robots_depleted` ve `max_steps` değerlerinden geçerli olanların listesi.
- `messages_sent`, `messages_lost`: gönderilen ve kaybolan mesaj sayısı.
- `weeds_total`, `weeds_sprayed`, `unnecessary_sprays`: toplam ot hücresi, ilaçlanan ot ve gereksiz ilaçlama sayıları.
- `weed_sprayed_ratio`: ilaçlanan ot oranı.
- `sprayed_area_ratio`: ilaçlanan hücrelerin tarlaya oranı.
- `path_length`: ajan başına yapılan hareket sayısı; `path_length_total` bunların toplamı.
- `completion_step`: tüm otların ilaçlandığı adım; ilaçlanmadıysa `None`.

## EPyMARL yardımcıları

Pelin'in `mappo_requirements.md` belgesindeki metotlar ince birer sarmalayıcı olarak eklendi:

- `get_obs()`, `get_obs_agent(i)`, `get_obs_size()`
- `get_state()`, `get_state_size()`
- `get_avail_actions()`, `get_avail_agent_actions(i)`, `get_total_actions()`
- `get_env_info()`: `state_shape`, `obs_shape`, `n_actions`, `n_agents` ve `episode_limit` anahtarlarını döndürür.

## Çizim

`render_mode="ansi"` (varsayılan) metin döndürür. Simgeler:
- `.` toprak, `c` ürün, `W` ot
- `x` ilaçlanmış ot, `s` gereksiz ilaçlanmış hücre
- `D` drone, `R` robot, `*` aynı hücrede birden fazla ajan

`render_mode="rgb_array"` hücre başına 8×8 piksellik `uint8` görüntü döndürür. Kaydetmek için örneğin `matplotlib.pyplot.imsave` kullanılabilir.

## Config alanları (`WeedEnvConfig`)

Tüm alanlar tek bir dataclass içindedir. `WeedEnvConfig.from_dict(d)` ile bir sözlükten (ileride YAML'dan) okunur; bilinmeyen alan verilirse hata verir.

| Alan | Varsayılan | Açıklama |
|---|---|---|
| `grid_path` | `None` | `.npy` tarla yolu. `None` ise rastgele tarla üretilir |
| `grid_size` | 16 | Rastgele tarlanın kenarı |
| `weed_density`, `crop_density` | 0.1, 0.2 | Rastgele tarlada ot ve ürün oranı |
| `crop_size` | `None` | `.npy` tarlayı kare kırpar (ör. 16 ya da 32) |
| `crop_origin` | `None` | Kırpma köşesi `(satır, sütun)`. `None` ise tohuma bağlı rastgele |
| `n_drones`, `n_robots` | 2, 2 | 1–4 arası |
| `max_steps` | 200 | Truncation sınırı |
| `drone_battery`, `robot_battery` | 200, 200 | Başlangıç bataryası |
| `drone_move_cost`, `robot_move_cost` | 1, 1 | Hareket başına maliyet |
| `stay_cost` | 0 | STAY maliyeti |
| `spray_cost` | 1 | İlaçlama maliyeti |
| `detection_radius` | 2 | Drone algılama yarıçapı (kare) |
| `obs_window` | 5 | Yerel pencere kenarı k (tek sayı) |
| `n_reports` | 5 | Robot gözlemindeki en yakın ot sayısı K |
| `state_map_size` | 16 | State haritasının kenarı S |
| `channel`, `channel_params` | `"perfect"`, `{}` | Kanal tipi ve parametreleri |
| `reward` | `RewardConfig()` | Ödül katsayıları (`marl/reward.py`) |

## Rastgelelik

Tüm rastgelelik ortamın kendi `np.random.Generator`'ından gelir. `reset(seed=s)` her seferinde bu üreteci yeniden kurar ve tarla, başlangıç konumları, dedektör ve kanal için ayrı alt üreteçler türetir (`Generator.spawn`).

Bu ayrımın sonucu: kanal tipi değiştiğinde tarla ve başlangıç konumları değişmez. Aynı tohumla deneyler yalnızca kanal bakımından farklılaşır.

## Bekleyen kararlar için öneriler

> **ÖNERİ — Pelin ile onaylanacak.** Aşağıdaki değerler `mappo_requirements.md` §9'daki "Pending Decisions" için seçildi. Hepsi config'den değiştirilebilir.

| Karar | Seçilen değer | Gerekçe |
|---|---|---|
| Gözlem boyutu | `6 + 4k² + 3K`. Varsayılan k = 5, K = 5 için 121 | Sabit uzunlukta. Drone ve robot aynı düzeni kullanır, robota özgü alanlar drone'da 0 kalır. Böylece MAPPO tek bir aktör ağını paylaşabilir |
| Global state | Ajan konum ve bataryaları + 3 kanallı S×S indirgenmiş harita + ilerleme. Varsayılan 781 | Sabit boyutlu ve tarla boyutundan bağımsız (S = 16). Gerçek ot dağılımını yalnızca kritik görür |
| Algılama menzili | `detection_radius = 2`, yani 5×5 kare | 64×64 tarlada tek drone'un adım başına %0,6'lık alanı görmesi, keşfi önemli bir görev haline getirir. Sema'nın dedektör ölçümlerine göre güncellenebilir |
| Batarya modeli | Hareket 1, STAY 0, ilaçlama 1. Kapasite 200 | Basit ve doğrusal. Bataryası biten ajan yalnızca STAY yapar; drone'un algılaması ve iletişimi sürer |
| Mesaj temsili | Her adımda her drone→robot bağlantısında bir `WeedReport`: taranan blok, ot hücreleri, güven | Bağlantı başına adımda tek mesaj olunca Gilbert-Elliott'un "patlama" uzunluğu doğrudan adım cinsinden yorumlanır |
| Bitiş koşulları | Termination: tüm otlar ilaçlandı ya da tüm robotların bataryası bitti. Truncation: max_steps | Görev başarısı ile zaman sınırı ayrı raporlanır; EPyMARL'ın bootstrapping için ihtiyaç duyduğu ayrım budur |
