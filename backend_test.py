"""Backend API tests for FLOWTOX_REGIME_01 strategy backtesting system."""

import requests
import sys
import time
from datetime import datetime

class BackendAPITester:
    def __init__(self, base_url="https://dev-factory-10.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.run_id = None
        self.job_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=120):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    json_data = response.json()
                    return success, json_data
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"   Response: {response.text[:200]}")
                except:
                    pass
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_instruments(self):
        """Test GET /api/instruments - should return all 6 instruments"""
        success, response = self.run_test(
            "GET /api/instruments",
            "GET",
            "api/instruments",
            200
        )
        if success and 'instruments' in response:
            instruments = response['instruments']
            print(f"   Found {len(instruments)} instruments")
            
            # Expected instruments with their specs
            expected = {
                'ES': {'tick_size': 0.25, 'point_value': 50.0},
                'MES': {'tick_size': 0.25, 'point_value': 5.0},
                'MNQ': {'tick_size': 0.25, 'point_value': 2.0},
                'M2K': {'tick_size': 0.10, 'point_value': 5.0},
                'MGC': {'tick_size': 0.10, 'point_value': 10.0},
                'MCL': {'tick_size': 0.01, 'point_value': 100.0},
            }
            
            all_correct = True
            for symbol, specs in expected.items():
                inst = next((i for i in instruments if i['symbol'] == symbol), None)
                if inst:
                    tick_ok = inst['tick_size'] == specs['tick_size']
                    pv_ok = inst['point_value'] == specs['point_value']
                    avail_ok = inst.get('available', False)
                    
                    status = "✓" if (tick_ok and pv_ok and avail_ok) else "✗"
                    print(f"   {status} {symbol}: tick={inst['tick_size']} (expect {specs['tick_size']}), pv={inst['point_value']} (expect {specs['point_value']}), available={avail_ok}")
                    
                    if not (tick_ok and pv_ok and avail_ok):
                        all_correct = False
                else:
                    print(f"   ✗ {symbol}: NOT FOUND")
                    all_correct = False
            
            return success and len(instruments) == 6 and all_correct
        return False

    def test_strategy_info(self):
        """Test GET /api/strategy/info"""
        success, response = self.run_test(
            "GET /api/strategy/info",
            "GET",
            "api/strategy/info",
            200
        )
        if success:
            has_name = 'name' in response
            has_params = 'param_defaults' in response
            has_space = 'param_space' in response
            has_criteria = 'acceptance_criteria' in response
            print(f"   Has name: {has_name}")
            print(f"   Has param_defaults: {has_params}")
            print(f"   Has param_space: {has_space}")
            print(f"   Has acceptance_criteria: {has_criteria}")
            return success and has_name and has_params and has_space and has_criteria
        return False

    def test_single_backtest_es(self):
        """Test POST /api/backtest/single for ES (regression check)"""
        print("\n⚠️  NOTE: First backtest after backend restart takes 60-90s for model warm-up")
        success, response = self.run_test(
            "POST /api/backtest/single (ES)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.5,
                "toxic_reversal_threshold": 0.5,
                "max_hold_bars": 15,
                "regime_exit_enabled": True
            },
            timeout=120  # generous timeout for first run
        )
        if success:
            # Check required fields
            has_metrics = 'metrics' in response
            has_checks = 'checks' in response
            has_model_info = 'model_info' in response
            has_charts = 'charts' in response
            has_trades = 'trades' in response
            has_run_id = 'run_id' in response
            
            print(f"   Has metrics: {has_metrics}")
            print(f"   Has checks: {has_checks}")
            print(f"   Has model_info: {has_model_info}")
            print(f"   Has charts: {has_charts}")
            print(f"   Has trades: {has_trades}")
            print(f"   Has run_id: {has_run_id}")
            
            if has_run_id:
                self.run_id = response['run_id']
                print(f"   Run ID: {self.run_id}")
            
            if has_metrics:
                metrics = response['metrics']
                print(f"   Total trades: {metrics.get('total_trades', 'N/A')}")
                print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A')}")
                print(f"   Max drawdown: {metrics.get('max_drawdown_pct', 'N/A')}")
            
            if has_checks:
                checks = response['checks']
                print(f"   Acceptance checks: {checks}")
                print(f"   ⚠️  NOTE: Strategy INTENTIONALLY FAILS acceptance criteria (expected behavior)")
            
            # Check chart series
            if has_charts:
                charts = response['charts']
                chart_keys = ['equity_series', 'drawdown_series', 'regime_series', 'vol_series', 'monthly_returns', 'pnl_histogram']
                for key in chart_keys:
                    has_key = key in charts
                    print(f"   Has {key}: {has_key}")
            
            # Verify ES point_value is 50.0
            pv_correct = False
            if has_model_info:
                model_info = response['model_info']
                pv = model_info.get('point_value')
                pv_correct = pv == 50.0
                print(f"   ES point_value: {pv} (expect 50.0) {'✓' if pv_correct else '✗'}")
            
            return success and has_metrics and has_checks and has_model_info and has_charts and has_trades and pv_correct
        return False

    def test_single_backtest_m2k(self):
        """Test POST /api/backtest/single for M2K (tick 0.10, pv 5.0)"""
        print("\n⚠️  NOTE: First run for M2K takes 60-90s for model fitting")
        success, response = self.run_test(
            "POST /api/backtest/single (M2K)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "M2K",
                "toxic_continuation_threshold": 0.5,
                "toxic_reversal_threshold": 0.5,
                "max_hold_bars": 15,
                "regime_exit_enabled": True
            },
            timeout=120
        )
        if success:
            has_metrics = 'metrics' in response
            has_checks = 'checks' in response
            has_model_info = 'model_info' in response
            has_charts = 'charts' in response
            has_trades = 'trades' in response
            
            print(f"   Has metrics: {has_metrics}")
            print(f"   Has model_info: {has_model_info}")
            print(f"   Has trades: {has_trades}")
            
            # Verify M2K specs
            tick_ok = pv_ok = test_days_ok = test_rows_ok = trades_ok = False
            if has_model_info:
                model_info = response['model_info']
                tick = model_info.get('tick_size')
                pv = model_info.get('point_value')
                test_days = model_info.get('num_test_days', 0)
                test_rows = model_info.get('test_rows', 0)
                
                tick_ok = tick == 0.10
                pv_ok = pv == 5.0
                test_days_ok = test_days > 0
                test_rows_ok = test_rows > 0
                
                print(f"   M2K tick_size: {tick} (expect 0.10) {'✓' if tick_ok else '✗'}")
                print(f"   M2K point_value: {pv} (expect 5.0) {'✓' if pv_ok else '✗'}")
                print(f"   num_test_days: {test_days} {'✓' if test_days_ok else '✗'}")
                print(f"   test_rows: {test_rows} {'✓' if test_rows_ok else '✗'}")
            
            if has_trades:
                trades = response['trades']
                trades_ok = len(trades) > 0
                print(f"   Trades count: {len(trades)} {'✓' if trades_ok else '✗'}")
            
            return success and tick_ok and pv_ok and test_days_ok and test_rows_ok and trades_ok
        return False

    def test_single_backtest_mcl(self):
        """Test POST /api/backtest/single for MCL (tick 0.01, pv 100.0, ~66 bars/day)"""
        print("\n⚠️  NOTE: First run for MCL takes 60-90s for model fitting")
        success, response = self.run_test(
            "POST /api/backtest/single (MCL)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "MCL",
                "toxic_continuation_threshold": 0.5,
                "toxic_reversal_threshold": 0.5,
                "max_hold_bars": 15,
                "regime_exit_enabled": True
            },
            timeout=120
        )
        if success:
            has_metrics = 'metrics' in response
            has_model_info = 'model_info' in response
            has_trades = 'trades' in response
            
            print(f"   Has metrics: {has_metrics}")
            print(f"   Has model_info: {has_model_info}")
            print(f"   Has trades: {has_trades}")
            
            # Verify MCL specs (different session length)
            tick_ok = pv_ok = test_days_ok = test_rows_ok = trades_ok = False
            if has_model_info:
                model_info = response['model_info']
                tick = model_info.get('tick_size')
                pv = model_info.get('point_value')
                test_days = model_info.get('num_test_days', 0)
                test_rows = model_info.get('test_rows', 0)
                
                tick_ok = tick == 0.01
                pv_ok = pv == 100.0
                test_days_ok = test_days > 0
                test_rows_ok = test_rows > 0
                
                print(f"   MCL tick_size: {tick} (expect 0.01) {'✓' if tick_ok else '✗'}")
                print(f"   MCL point_value: {pv} (expect 100.0) {'✓' if pv_ok else '✗'}")
                print(f"   num_test_days: {test_days} {'✓' if test_days_ok else '✗'}")
                print(f"   test_rows: {test_rows} {'✓' if test_rows_ok else '✗'}")
            
            if has_trades:
                trades = response['trades']
                trades_ok = len(trades) > 0
                print(f"   Trades count: {len(trades)} {'✓' if trades_ok else '✗'}")
            
            return success and tick_ok and pv_ok and test_days_ok and test_rows_ok and trades_ok
        return False

    def test_optimize_start(self):
        """Test POST /api/optimize/start"""
        success, response = self.run_test(
            "POST /api/optimize/start",
            "POST",
            "api/optimize/start",
            200,
            data={"symbol": "ES"}
        )
        if success and 'job_id' in response:
            self.job_id = response['job_id']
            print(f"   Job ID: {self.job_id}")
            print(f"   Status: {response.get('status', 'N/A')}")
            return True
        return False

    def test_optimize_status(self):
        """Test GET /api/optimize/status/{job_id} (poll for completion)"""
        if not self.job_id:
            print("❌ No job_id available, skipping optimize status test")
            return False
        
        print(f"\n⚠️  NOTE: Optimization takes ~3-4 minutes, polling for up to 6 minutes...")
        max_wait = 360  # 6 minutes
        poll_interval = 3  # 3 seconds
        elapsed = 0
        
        while elapsed < max_wait:
            success, response = self.run_test(
                f"GET /api/optimize/status/{self.job_id}",
                "GET",
                f"api/optimize/status/{self.job_id}",
                200,
                timeout=10
            )
            
            if not success:
                return False
            
            status = response.get('status', 'unknown')
            pct = response.get('pct', 0)
            print(f"   Status: {status}, Progress: {pct}%")
            
            if status == 'done':
                print(f"   ✅ Optimization completed!")
                has_result = 'result' in response
                print(f"   Has result: {has_result}")
                
                if has_result:
                    result = response['result']
                    has_best_params = 'best_params' in result
                    has_wf_sharpe = 'walk_forward_sharpe' in result
                    has_sensitivity = 'sensitivity' in result
                    has_metrics = 'metrics' in result
                    has_checks = 'checks' in result
                    
                    print(f"   Has best_params: {has_best_params}")
                    print(f"   Has walk_forward_sharpe: {has_wf_sharpe}")
                    print(f"   Has sensitivity: {has_sensitivity}")
                    print(f"   Has metrics: {has_metrics}")
                    print(f"   Has checks: {has_checks}")
                    
                    if has_best_params:
                        print(f"   Best params: {result['best_params']}")
                    if has_wf_sharpe:
                        print(f"   Walk-forward Sharpe: {result['walk_forward_sharpe']}")
                    
                    # Store run_id for download tests
                    if 'run_id' in result:
                        self.run_id = result['run_id']
                    
                    return has_result and has_best_params and has_wf_sharpe and has_sensitivity
                return False
            
            elif status == 'failed':
                error = response.get('error', 'unknown error')
                print(f"   ❌ Optimization failed: {error}")
                return False
            
            # Still running, wait and poll again
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        print(f"   ❌ Optimization did not complete within {max_wait}s")
        return False

    def test_get_run(self):
        """Test GET /api/runs/{run_id}"""
        if not self.run_id:
            print("❌ No run_id available, skipping get run test")
            return False
        
        success, response = self.run_test(
            f"GET /api/runs/{self.run_id}",
            "GET",
            f"api/runs/{self.run_id}",
            200
        )
        if success:
            has_metrics = 'metrics' in response
            has_trades = 'trades' in response
            has_charts = 'charts' in response
            print(f"   Has metrics: {has_metrics}")
            print(f"   Has trades: {has_trades}")
            print(f"   Has charts: {has_charts}")
            return success and has_metrics
        return False

    def test_download_files(self):
        """Test GET /api/runs/{run_id}/download/{kind}"""
        if not self.run_id:
            print("❌ No run_id available, skipping download tests")
            return False
        
        kinds = ['metrics', 'trades', 'equity']
        all_success = True
        
        for kind in kinds:
            success, _ = self.run_test(
                f"GET /api/runs/{self.run_id}/download/{kind}",
                "GET",
                f"api/runs/{self.run_id}/download/{kind}",
                200
            )
            if not success:
                all_success = False
        
        return all_success

    def test_pine_script(self):
        """Test GET /api/pine-script - should return text/plain starting with '//@version=6'"""
        url = f"{self.base_url}/api/pine-script"
        self.tests_run += 1
        print(f"\n🔍 Testing GET /api/pine-script...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            success = response.status_code == 200
            
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                is_text = 'text/plain' in content_type
                print(f"   Content-Type: {content_type} {'✓' if is_text else '✗'}")
                
                # Check content starts with '//@version=6'
                text = response.text
                starts_correct = text.startswith('//@version=6')
                print(f"   Starts with '//@version=6': {starts_correct} {'✓' if starts_correct else '✗'}")
                print(f"   First 100 chars: {text[:100]}")
                
                return success and is_text and starts_correct
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False

    def test_drawdown_anchored(self):
        """Test drawdown_method='anchored' - should return both drawdown values, max_drawdown == max_drawdown_anchored"""
        print("\n⚠️  NOTE: Testing dual drawdown with method='anchored'")
        success, response = self.run_test(
            "POST /api/backtest/single (drawdown_method=anchored)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored"
            },
            timeout=120
        )
        
        if success and 'metrics' in response:
            metrics = response['metrics']
            
            # Check both drawdown values are present
            has_anchored = 'max_drawdown_anchored' in metrics
            has_spec = 'max_drawdown_spec' in metrics
            has_method = 'drawdown_method' in metrics
            has_max_dd = 'max_drawdown' in metrics
            
            print(f"   Has max_drawdown_anchored: {has_anchored}")
            print(f"   Has max_drawdown_spec: {has_spec}")
            print(f"   Has drawdown_method: {has_method}")
            print(f"   Has max_drawdown: {has_max_dd}")
            
            if has_anchored and has_spec and has_method and has_max_dd:
                anchored = metrics['max_drawdown_anchored']
                spec = metrics['max_drawdown_spec']
                method = metrics['drawdown_method']
                max_dd = metrics['max_drawdown']
                
                print(f"   max_drawdown_anchored: {anchored}")
                print(f"   max_drawdown_spec: {spec}")
                print(f"   drawdown_method: {method}")
                print(f"   max_drawdown: {max_dd}")
                
                # Verify method is 'anchored'
                method_correct = method == 'anchored'
                print(f"   drawdown_method == 'anchored': {method_correct} {'✓' if method_correct else '✗'}")
                
                # Verify max_drawdown == max_drawdown_anchored
                max_dd_correct = abs(max_dd - anchored) < 0.0001
                print(f"   max_drawdown == max_drawdown_anchored: {max_dd_correct} {'✓' if max_dd_correct else '✗'}")
                
                # Verify anchored and spec are different (they should be)
                different = abs(anchored - spec) > 0.001
                print(f"   anchored != spec: {different} {'✓' if different else '✗'}")
                
                return success and method_correct and max_dd_correct and different
            
        return False

    def test_drawdown_spec(self):
        """Test drawdown_method='spec' - should return max_drawdown == max_drawdown_spec (different from anchored)"""
        print("\n⚠️  NOTE: Testing dual drawdown with method='spec'")
        success, response = self.run_test(
            "POST /api/backtest/single (drawdown_method=spec)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "spec"
            },
            timeout=120
        )
        
        if success and 'metrics' in response:
            metrics = response['metrics']
            
            has_anchored = 'max_drawdown_anchored' in metrics
            has_spec = 'max_drawdown_spec' in metrics
            has_method = 'drawdown_method' in metrics
            has_max_dd = 'max_drawdown' in metrics
            
            if has_anchored and has_spec and has_method and has_max_dd:
                anchored = metrics['max_drawdown_anchored']
                spec = metrics['max_drawdown_spec']
                method = metrics['drawdown_method']
                max_dd = metrics['max_drawdown']
                
                print(f"   max_drawdown_anchored: {anchored}")
                print(f"   max_drawdown_spec: {spec}")
                print(f"   drawdown_method: {method}")
                print(f"   max_drawdown: {max_dd}")
                
                # Verify method is 'spec'
                method_correct = method == 'spec'
                print(f"   drawdown_method == 'spec': {method_correct} {'✓' if method_correct else '✗'}")
                
                # Verify max_drawdown == max_drawdown_spec
                max_dd_correct = abs(max_dd - spec) < 0.0001
                print(f"   max_drawdown == max_drawdown_spec: {max_dd_correct} {'✓' if max_dd_correct else '✗'}")
                
                # Verify anchored and spec are different
                different = abs(anchored - spec) > 0.001
                print(f"   anchored != spec: {different} {'✓' if different else '✗'}")
                
                return success and method_correct and max_dd_correct and different
            
        return False

    def test_diagnostics_single(self):
        """Test single backtest diagnostics - should have PSR, sharpe_ci_low, sharpe_ci_high, bootstrap_p_value"""
        print("\n⚠️  NOTE: Testing robustness diagnostics for single backtest")
        success, response = self.run_test(
            "POST /api/backtest/single (diagnostics check)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored"
            },
            timeout=120
        )
        
        if success and 'diagnostics' in response:
            diag = response['diagnostics']
            
            # Check required fields for single backtest
            has_psr = 'psr' in diag
            has_ci_low = 'sharpe_ci_low' in diag
            has_ci_high = 'sharpe_ci_high' in diag
            has_boot_p = 'bootstrap_p_value' in diag
            
            print(f"   Has psr: {has_psr}")
            print(f"   Has sharpe_ci_low: {has_ci_low}")
            print(f"   Has sharpe_ci_high: {has_ci_high}")
            print(f"   Has bootstrap_p_value: {has_boot_p}")
            
            if has_psr and has_ci_low and has_ci_high and has_boot_p:
                psr = diag['psr']
                ci_low = diag['sharpe_ci_low']
                ci_high = diag['sharpe_ci_high']
                boot_p = diag['bootstrap_p_value']
                
                print(f"   psr: {psr}")
                print(f"   sharpe_ci_low: {ci_low}")
                print(f"   sharpe_ci_high: {ci_high}")
                print(f"   bootstrap_p_value: {boot_p}")
                
                # Verify values are numeric (not null)
                psr_ok = psr is not None and isinstance(psr, (int, float))
                ci_low_ok = ci_low is not None and isinstance(ci_low, (int, float))
                ci_high_ok = ci_high is not None and isinstance(ci_high, (int, float))
                boot_p_ok = boot_p is not None and isinstance(boot_p, (int, float))
                
                print(f"   psr is numeric: {psr_ok} {'✓' if psr_ok else '✗'}")
                print(f"   sharpe_ci_low is numeric: {ci_low_ok} {'✓' if ci_low_ok else '✗'}")
                print(f"   sharpe_ci_high is numeric: {ci_high_ok} {'✓' if ci_high_ok else '✗'}")
                print(f"   bootstrap_p_value is numeric: {boot_p_ok} {'✓' if boot_p_ok else '✗'}")
                
                # Check PBO/DSR should be null for single runs
                pbo = diag.get('pbo')
                dsr = diag.get('dsr')
                print(f"   pbo (should be null for single): {pbo}")
                print(f"   dsr (should be null for single): {dsr}")
                
                return success and psr_ok and ci_low_ok and ci_high_ok and boot_p_ok
            
        return False

    def test_hmm_backtest(self):
        """Test POST /api/backtest/single with regime_model='hmm' - NEW FEATURE"""
        print("\n⚠️  NOTE: Testing HMM regime model (first call may take ~30s for fitting)")
        success, response = self.run_test(
            "POST /api/backtest/single (regime_model=hmm)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored",
                "regime_model": "hmm"
            },
            timeout=120
        )
        
        if success:
            # Check regime_model in payload
            regime_model = response.get('regime_model')
            regime_model_ok = regime_model == 'hmm'
            print(f"   payload.regime_model: {regime_model} {'✓' if regime_model_ok else '✗'}")
            
            # Check model_info.regime_model
            model_info = response.get('model_info', {})
            mi_regime_model = model_info.get('regime_model')
            mi_regime_ok = mi_regime_model == 'hmm'
            print(f"   model_info.regime_model: {mi_regime_model} {'✓' if mi_regime_ok else '✗'}")
            
            # Check transition_matrix exists and is 3x3
            transition_matrix = model_info.get('transition_matrix')
            has_trans = transition_matrix is not None
            print(f"   Has transition_matrix: {has_trans}")
            
            trans_ok = False
            if has_trans:
                is_list = isinstance(transition_matrix, list)
                is_3x3 = is_list and len(transition_matrix) == 3 and all(len(row) == 3 for row in transition_matrix)
                print(f"   transition_matrix is 3x3: {is_3x3} {'✓' if is_3x3 else '✗'}")
                
                if is_3x3:
                    # Check each row sums to ~1.0
                    row_sums = [sum(row) for row in transition_matrix]
                    sums_ok = all(abs(s - 1.0) < 0.01 for s in row_sums)
                    print(f"   Row sums: {[round(s, 3) for s in row_sums]} {'✓' if sums_ok else '✗'}")
                    
                    # Print the matrix
                    print(f"   Transition matrix:")
                    for i, row in enumerate(transition_matrix):
                        print(f"     [{', '.join(f'{v:.3f}' for v in row)}]")
                    
                    trans_ok = sums_ok
            
            # Check trades exist
            trades = response.get('trades', [])
            has_trades = len(trades) > 0
            print(f"   Has trades: {has_trades} (count: {len(trades)}) {'✓' if has_trades else '✗'}")
            
            # Check metrics
            metrics = response.get('metrics', {})
            has_sharpe = 'sharpe_ratio' in metrics
            sharpe = metrics.get('sharpe_ratio')
            print(f"   Has sharpe_ratio: {has_sharpe}")
            if has_sharpe:
                print(f"   Sharpe ratio: {sharpe}")
            
            # Check diagnostics still present
            has_diagnostics = 'diagnostics' in response
            print(f"   Has diagnostics: {has_diagnostics} {'✓' if has_diagnostics else '✗'}")
            
            return success and regime_model_ok and mi_regime_ok and trans_ok and has_trades and has_diagnostics
        
        return False

    def test_gmm_backtest(self):
        """Test POST /api/backtest/single with regime_model='gmm' - verify no transition matrix"""
        print("\n⚠️  NOTE: Testing GMM regime model (should have no transition matrix)")
        success, response = self.run_test(
            "POST /api/backtest/single (regime_model=gmm)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored",
                "regime_model": "gmm"
            },
            timeout=120
        )
        
        if success:
            # Check regime_model in payload
            regime_model = response.get('regime_model')
            regime_model_ok = regime_model == 'gmm'
            print(f"   payload.regime_model: {regime_model} {'✓' if regime_model_ok else '✗'}")
            
            # Check model_info.regime_model
            model_info = response.get('model_info', {})
            mi_regime_model = model_info.get('regime_model')
            mi_regime_ok = mi_regime_model == 'gmm'
            print(f"   model_info.regime_model: {mi_regime_model} {'✓' if mi_regime_ok else '✗'}")
            
            # Check transition_matrix is null
            transition_matrix = model_info.get('transition_matrix')
            trans_null = transition_matrix is None
            print(f"   transition_matrix is null: {trans_null} {'✓' if trans_null else '✗'}")
            
            # Check trades exist
            trades = response.get('trades', [])
            has_trades = len(trades) > 0
            print(f"   Has trades: {has_trades} (count: {len(trades)}) {'✓' if has_trades else '✗'}")
            
            # Check metrics
            metrics = response.get('metrics', {})
            sharpe = metrics.get('sharpe_ratio')
            print(f"   Sharpe ratio: {sharpe}")
            
            return success and regime_model_ok and mi_regime_ok and trans_null and has_trades
        
        return False

    def test_hmm_gmm_different_results(self):
        """Test that HMM and GMM produce different results (regime model actually changes results)"""
        print("\n⚠️  NOTE: Testing that HMM and GMM produce different Sharpe ratios")
        
        # Run HMM
        success_hmm, response_hmm = self.run_test(
            "POST /api/backtest/single (HMM for comparison)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored",
                "regime_model": "hmm"
            },
            timeout=120
        )
        
        # Run GMM
        success_gmm, response_gmm = self.run_test(
            "POST /api/backtest/single (GMM for comparison)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "drawdown_method": "anchored",
                "regime_model": "gmm"
            },
            timeout=120
        )
        
        if success_hmm and success_gmm:
            sharpe_hmm = response_hmm.get('metrics', {}).get('sharpe_ratio')
            sharpe_gmm = response_gmm.get('metrics', {}).get('sharpe_ratio')
            trades_hmm = len(response_hmm.get('trades', []))
            trades_gmm = len(response_gmm.get('trades', []))
            
            print(f"   HMM Sharpe: {sharpe_hmm}, Trades: {trades_hmm}")
            print(f"   GMM Sharpe: {sharpe_gmm}, Trades: {trades_gmm}")
            
            # Check that results are different
            sharpe_different = sharpe_hmm is not None and sharpe_gmm is not None and abs(sharpe_hmm - sharpe_gmm) > 0.01
            trades_different = trades_hmm != trades_gmm
            
            print(f"   Sharpe ratios different: {sharpe_different} {'✓' if sharpe_different else '✗'}")
            print(f"   Trade counts different: {trades_different} {'✓' if trades_different else '✗'}")
            
            return success_hmm and success_gmm and (sharpe_different or trades_different)
        
        return False

    def test_hmm_cache_speed(self):
        """Test that second HMM call for same symbol is fast (<5s) due to caching"""
        print("\n⚠️  NOTE: Testing HMM caching (second call should be fast)")
        
        # First call (may be slow if not cached)
        start1 = time.time()
        success1, response1 = self.run_test(
            "POST /api/backtest/single (HMM first call)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.55,
                "toxic_reversal_threshold": 0.55,
                "max_hold_bars": 15,
                "regime_exit_enabled": True,
                "regime_model": "hmm"
            },
            timeout=120
        )
        elapsed1 = time.time() - start1
        print(f"   First call took: {elapsed1:.2f}s")
        
        # Second call (should be fast)
        start2 = time.time()
        success2, response2 = self.run_test(
            "POST /api/backtest/single (HMM second call)",
            "POST",
            "api/backtest/single",
            200,
            data={
                "symbol": "ES",
                "toxic_continuation_threshold": 0.60,  # different params
                "toxic_reversal_threshold": 0.60,
                "max_hold_bars": 20,
                "regime_exit_enabled": True,
                "regime_model": "hmm"
            },
            timeout=120
        )
        elapsed2 = time.time() - start2
        print(f"   Second call took: {elapsed2:.2f}s")
        
        # Check second call is fast
        is_fast = elapsed2 < 5.0
        print(f"   Second call < 5s: {is_fast} {'✓' if is_fast else '✗'}")
        
        return success1 and success2 and is_fast

def main():
    print("=" * 80)
    print("FLOWTOX_REGIME_01 Backend API Test Suite - HMM FEATURE")
    print("=" * 80)
    
    tester = BackendAPITester()
    
    # Test NEW HMM feature
    print("\n" + "=" * 80)
    print("PHASE 1: NEW FEATURE - HMM Regime Model")
    print("=" * 80)
    tester.test_hmm_backtest()
    tester.test_gmm_backtest()
    tester.test_hmm_gmm_different_results()
    tester.test_hmm_cache_speed()
    
    # Test regression: drawdown_method and diagnostics still work
    print("\n" + "=" * 80)
    print("PHASE 2: REGRESSION - Drawdown & Diagnostics")
    print("=" * 80)
    tester.test_drawdown_anchored()
    tester.test_diagnostics_single()
    
    # Test basic endpoints (regression)
    print("\n" + "=" * 80)
    print("PHASE 3: REGRESSION - Basic Endpoints")
    print("=" * 80)
    tester.test_instruments()
    tester.test_strategy_info()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {tester.tests_run}")
    print(f"Tests passed: {tester.tests_passed}")
    print(f"Tests failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success rate: {(tester.tests_passed / tester.tests_run * 100):.1f}%")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
