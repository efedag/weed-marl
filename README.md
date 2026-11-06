# Otonom Tarım için Çoklu Ajan Sistemleri (Grup 4)

Bitirme projesi. Ekip: Efe İbrahim Dağ, Sema Nur Özyılmaz, Pelin Konak.

## Proje özeti

Drone ve yer robotu ekipleri için PettingZoo tabanlı, çok ajanlı bir simülatör geliştiriyoruz. Ana senaryo yabancı ot ilaçlama, ana araştırma konusu ise mesaj kaybı altında görev dağılımıdır.

## Klasör yapısı ve sorumlular

| Klasör | İçerik | Sorumlu |
|---|---|---|
| `data/` | WeedMap'ten üretilen ızgara tarlalar (`.npy`) | Sema Nur |
| `sim/` | PettingZoo ortamı, ajanlar | Efe İbrahim |
| `sim/comms/` | Mesaj kanalı: Bernoulli ve Gilbert-Elliott | Efe İbrahim |
| `detector/` | Simüle dedektör | Sema Nur |
| `baselines/` | Sabit rota, tek robot | Pelin |
| `baselines/auction/` | Sıralı açık artırma ve CBBA | Efe İbrahim |
| `marl/` | EPyMARL ayarları, MAPPO/QMIX betikleri | Pelin |
| `experiments/` | YAML ayarları, deney koşturucu | Sema Nur |
| `analysis/` | Tablo/grafik betikleri | Sema Nur |
| `paper/` | IEEE LaTeX şablonu (Overleaf) | Üç kişi |
| `tests/` | Birim testleri | Herkes |

## Kurulum

Python 3.11 gerekir. `requirements.txt` içindeki `numpy`, `pandas` ve `matplotlib` sürümleri Python 3.10'u desteklemez.

Önce depoyu klonlayın:

```bash
git clone https://github.com/efedag/weed-marl.git
cd weed-marl
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell betik çalıştırmayı engellerse, etkinleştirmeden önce şunu çalıştırın. Bu ayar yalnızca açık olan pencere için geçerlidir:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Testleri çalıştırma

Sanal ortam etkinken depo kök klasöründe:

```bash
pytest
```

### Sorun giderme

Antivirüs veya kurum ağı HTTPS trafiğini tarıyorsa `pip` `CERTIFICATE_VERIFY_FAILED` hatası verebilir. `--trusted-host` ile doğrulamayı kapatmak yerine işletim sisteminin sertifika deposunu kullanın:

```bash
python -m pip install --use-feature=truststore --upgrade pip
```

Güncel `pip` bu depoyu zaten varsayılan olarak kullanır, bundan sonra `pip install -r requirements.txt` normal şekilde çalışır.

## Örneği çalıştır

`sim/example_parallel_env.py`, 8×8 ızgarada 1 drone ve 1 yer robotu içeren örnek bir `ParallelEnv` ortamıdır. Aşağıdaki komut rastgele eylemlerle bir bölüm koşar ve adım sayısını yazar:

```bash
python -m sim.example_parallel_env
```

## Çalışma kuralları

- `main` dalına doğrudan push yapılmaz.
- Her değişiklik PR ile gelir ve başka bir ekip üyesinin onayı gerekir.
- Her görev için bir issue açılır.
