# Custom Pong

## Install

```bash
python -m pip install pygame-ce
```

## Run

```bash
python main.py --mode normal
```

Modes: `normal` | `gravity` | `teleport`

## Controls

`Up` move up  
`Down` move down  
`R` reset  
`1 2 3` switch mode  
`Esc` quit

## Test

```bash
python -m unittest discover -s tests -v
```

## Headless

Use `render_mode=None` in `PongEnv(...)`.
