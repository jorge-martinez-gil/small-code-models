@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Windows end-to-end automation for small-code-model clone detection.
rem Run from the repository root:
rem   run_everything.bat
rem
rem Common overrides:
rem   set RUN_BENCHMARKS=0
rem   set MODELS=codebert graphcodebert unixcoder
rem   set BENCHMARKS=bcb poj104
rem   set SAMPLE_PCT=100.0
rem   run_everything.bat

cd /d "%~dp0"

if not defined PYTHON_CMD (
  if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
)
if not defined PYTHON_CMD (
  if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  )
)
if not defined PYTHON_CMD (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined DATASETS_ROOT set "DATASETS_ROOT=datasets"
if not defined RESULTS_ROOT set "RESULTS_ROOT=results"
if not defined HF_CACHE_DIR set "HF_CACHE_DIR=.hf_cache"

if not defined INSTALL_DEPS set "INSTALL_DEPS=1"
if not defined INSTALL_DEV set "INSTALL_DEV=0"
if not defined RUN_TESTS set "RUN_TESTS=0"
if not defined AUTO_DOWNLOAD_DATASETS set "AUTO_DOWNLOAD_DATASETS=1"
if not defined NORMALIZE_LOCAL_DATASETS set "NORMALIZE_LOCAL_DATASETS=1"
if not defined OVERWRITE_DATASETS set "OVERWRITE_DATASETS=0"
if not defined INSPECT_DATASETS set "INSPECT_DATASETS=1"
if not defined PREPARE_PROBLEM_DATASETS set "PREPARE_PROBLEM_DATASETS=0"
if not defined RUN_BENCHMARKS set "RUN_BENCHMARKS=1"
if not defined RUN_COMPARISONS set "RUN_COMPARISONS=1"

if not defined DEFAULT_MODELS (
  set "DEFAULT_MODELS=codebert graphcodebert unixcoder codet5 codet5_small codet5p_220m"
  set "DEFAULT_MODELS=!DEFAULT_MODELS! codeberta_small codegpt_py codegpt_java cotext_1_cc cotext_2_cc"
)
if not defined MODELS set "MODELS=%DEFAULT_MODELS%"
if not defined BENCHMARKS set "BENCHMARKS=bcb poj104 gcj karnalim poolc codenet semanticclonebench gptclonebench clcdsa"

if not defined EPOCHS set "EPOCHS=3"
if not defined SEED set "SEED=42"
if not defined SAMPLE_PCT set "SAMPLE_PCT=1.0"
if not defined MAX_LENGTH set "MAX_LENGTH=512"
if not defined TRAIN_BATCH_SIZE set "TRAIN_BATCH_SIZE=8"
if not defined EVAL_BATCH_SIZE set "EVAL_BATCH_SIZE=8"
if not defined BOOTSTRAP_RESAMPLES set "BOOTSTRAP_RESAMPLES=1000"
if not defined STRICT_DATA set "STRICT_DATA=1"
if not defined FP16 set "FP16=0"

if not defined PROBLEM_DATASETS set "PROBLEM_DATASETS=codenet clcdsa"
if not defined PROBLEM_SOURCE_ROOT set "PROBLEM_SOURCE_ROOT=problem_sources"
if not defined NEGATIVE_RATIO set "NEGATIVE_RATIO=1.0"
if not defined MAX_FILES_PER_PROBLEM set "MAX_FILES_PER_PROBLEM=50"

if not defined COMPARE_BASELINE set "COMPARE_BASELINE=codebert"
if not defined COMPARE_CANDIDATES set "COMPARE_CANDIDATES=graphcodebert unixcoder codet5_small"

echo == Environment ==
echo Repository: %CD%
echo Python command: %PYTHON_CMD%
call %PYTHON_CMD% --version
if errorlevel 1 goto :python_error

if not exist "%DATASETS_ROOT%" mkdir "%DATASETS_ROOT%"
if not exist "%RESULTS_ROOT%" mkdir "%RESULTS_ROOT%"
if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%"

if "%INSTALL_DEPS%"=="1" (
  echo == Install dependencies ==
  call %PYTHON_CMD% -m pip install --upgrade pip
  if errorlevel 1 goto :fail

  if "%INSTALL_DEV%"=="1" (
    call %PYTHON_CMD% -m pip install -e ".[dev]"
  ) else (
    call %PYTHON_CMD% -m pip install -e .
  )
  if errorlevel 1 goto :fail
)

echo == Registry ==
call %PYTHON_CMD% scripts\run_clone_experiment.py --list_models
if errorlevel 1 goto :fail
call %PYTHON_CMD% scripts\run_clone_experiment.py --list_benchmarks
if errorlevel 1 goto :fail

if "%AUTO_DOWNLOAD_DATASETS%"=="1" (
  echo == Download automatic datasets ==
  set "DOWNLOAD_EXISTING_ARG=--skip_existing"
  if "%OVERWRITE_DATASETS%"=="1" set "DOWNLOAD_EXISTING_ARG=--overwrite"
  call %PYTHON_CMD% scripts\download_datasets.py ^
    --dataset all ^
    --output_root "%DATASETS_ROOT%" ^
    --hf_cache_dir "%HF_CACHE_DIR%" ^
    !DOWNLOAD_EXISTING_ARG!
  if errorlevel 1 goto :fail
)

if "%RUN_TESTS%"=="1" (
  echo == Tests ==
  call %PYTHON_CMD% -m pytest tests -q
  if errorlevel 1 goto :fail
)

if "%NORMALIZE_LOCAL_DATASETS%"=="1" (
  echo == Normalize local datasets ==
  set "NORMALIZE_OVERWRITE_ARG="
  if "%OVERWRITE_DATASETS%"=="1" set "NORMALIZE_OVERWRITE_ARG=--overwrite"
  call %PYTHON_CMD% scripts\normalize_local_datasets.py ^
    --input_root "%DATASETS_ROOT%" ^
    --output_root "%DATASETS_ROOT%" ^
    --dataset all ^
    !NORMALIZE_OVERWRITE_ARG!
  if errorlevel 1 goto :fail
)

if "%PREPARE_PROBLEM_DATASETS%"=="1" (
  echo == Prepare problem-directory datasets ==
  for %%D in (%PROBLEM_DATASETS%) do (
    set "SOURCE_DIR=%PROBLEM_SOURCE_ROOT%\%%D"
    set "OUTPUT_DIR=%DATASETS_ROOT%\%%D"
    if exist "!SOURCE_DIR!\" (
      call %PYTHON_CMD% scripts\prepare_pair_dataset.py ^
        --source_dir "!SOURCE_DIR!" ^
        --output_dir "!OUTPUT_DIR!" ^
        --negative_ratio "%NEGATIVE_RATIO%" ^
        --seed "%SEED%" ^
        --max_files_per_problem "%MAX_FILES_PER_PROBLEM%" ^
        --split_strategy problem
      if errorlevel 1 goto :fail
    ) else (
      echo [SKIP] %%D: missing source directory "!SOURCE_DIR!"
    )
  )
)

if "%INSPECT_DATASETS%"=="1" (
  echo == Dataset diagnostics ==
  set "STRICT_ARG="
  if "%STRICT_DATA%"=="1" set "STRICT_ARG=--strict_data"
  for %%B in (%BENCHMARKS%) do (
    set "DATA_DIR=%DATASETS_ROOT%\%%B"
    call :missing_normalized_dataset_files "!DATA_DIR!"
    if not defined MISSING_FILES (
      call %PYTHON_CMD% scripts\inspect_dataset.py "!DATA_DIR!" ^
        !STRICT_ARG! ^
        --output "!DATA_DIR!\diagnostics.json"
      if errorlevel 1 goto :fail
    ) else (
      echo [SKIP] diagnostics for %%B: missing !MISSING_FILES!
    )
  )
)

if "%RUN_BENCHMARKS%"=="1" (
  echo == Benchmark matrix ==
  set "STATUS_FILE=%RESULTS_ROOT%\run_status.tsv"
  > "!STATUS_FILE!" echo model	benchmark	status	output_dir

  set "STRICT_ARG="
  if "%STRICT_DATA%"=="1" set "STRICT_ARG=--strict_data"
  set "FP16_ARG="
  if "%FP16%"=="1" set "FP16_ARG=--fp16"

  for %%B in (%BENCHMARKS%) do (
    set "DATA_DIR=%DATASETS_ROOT%\%%B"
    call :missing_normalized_dataset_files "!DATA_DIR!"
    if not defined MISSING_FILES (
      for %%M in (%MODELS%) do (
        set "OUTPUT_DIR=%RESULTS_ROOT%\%%M_%%B"
        echo [RUN] model=%%M benchmark=%%B
        call %PYTHON_CMD% scripts\run_clone_experiment.py ^
          --model "%%M" ^
          --benchmark "%%B" ^
          --data_dir "!DATA_DIR!" ^
          --output_dir "!OUTPUT_DIR!" ^
          --sample_pct "%SAMPLE_PCT%" ^
          --epochs "%EPOCHS%" ^
          --seed "%SEED%" ^
          --max_length "%MAX_LENGTH%" ^
          --train_batch_size "%TRAIN_BATCH_SIZE%" ^
          --eval_batch_size "%EVAL_BATCH_SIZE%" ^
          --bootstrap_resamples "%BOOTSTRAP_RESAMPLES%" ^
          !STRICT_ARG! ^
          !FP16_ARG!
        if errorlevel 1 (
          >> "!STATUS_FILE!" echo %%M	%%B	FAIL	!OUTPUT_DIR!
        ) else (
          >> "!STATUS_FILE!" echo %%M	%%B	OK	!OUTPUT_DIR!
        )
      )
    ) else (
      echo [SKIP] benchmark=%%B: missing !MISSING_FILES!
    )
  )

  echo == Summaries ==
  call %PYTHON_CMD% scripts\summarize_results.py "%RESULTS_ROOT%"
  if errorlevel 1 goto :fail
)

if "%RUN_COMPARISONS%"=="1" (
  echo == Pairwise comparisons ==
  set "COMPARISONS_DIR=%RESULTS_ROOT%\comparisons"
  if not exist "!COMPARISONS_DIR!" mkdir "!COMPARISONS_DIR!"
  for %%B in (%BENCHMARKS%) do (
    set "BASELINE_FILE=%RESULTS_ROOT%\%COMPARE_BASELINE%_%%B\predictions.jsonl"
    if exist "!BASELINE_FILE!" (
      for %%C in (%COMPARE_CANDIDATES%) do (
        set "CANDIDATE_FILE=%RESULTS_ROOT%\%%C_%%B\predictions.jsonl"
        if exist "!CANDIDATE_FILE!" (
          call %PYTHON_CMD% scripts\compare_predictions.py ^
            "!BASELINE_FILE!" ^
            "!CANDIDATE_FILE!" ^
            --metric f1 ^
            --bootstrap_resamples "%BOOTSTRAP_RESAMPLES%" ^
            --seed "%SEED%" ^
            --output "!COMPARISONS_DIR!\%%C_vs_%COMPARE_BASELINE%_%%B.json"
          if errorlevel 1 goto :fail
        ) else (
          echo [SKIP] %%C vs %COMPARE_BASELINE% on %%B: missing predictions
        )
      )
    ) else (
      echo [SKIP] comparisons for %%B: missing "!BASELINE_FILE!"
    )
  )
)

echo == Done ==
echo Datasets: %DATASETS_ROOT%
echo Results:  %RESULTS_ROOT%
exit /b 0

:missing_normalized_dataset_files
set "MISSING_FILES="
for %%F in (data.jsonl train.txt valid.txt test.txt) do (
  if not exist "%~1\%%F" (
    if defined MISSING_FILES (
      set "MISSING_FILES=!MISSING_FILES!, %~1\%%F"
    ) else (
      set "MISSING_FILES=%~1\%%F"
    )
  )
)
exit /b 0

:python_error
echo.
echo Python was not found. Set PYTHON_CMD before running this script, for example:
echo   set PYTHON_CMD=C:\Path\To\python.exe
exit /b 1

:fail
echo.
echo Automation failed. See the command output above for the failing step.
exit /b 1
