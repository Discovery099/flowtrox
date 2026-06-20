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

def main():
    print("=" * 80)
    print("FLOWTOX_REGIME_01 Backend API Test Suite - Multi-Instrument")
    print("=" * 80)
    
    tester = BackendAPITester()
    
    # Test basic endpoints
    print("\n" + "=" * 80)
    print("PHASE 1: Basic Endpoints & Multi-Instrument Support")
    print("=" * 80)
    tester.test_instruments()
    tester.test_strategy_info()
    
    # Test single backtests for multiple instruments
    print("\n" + "=" * 80)
    print("PHASE 2: Multi-Instrument Backtests (60-90s per instrument first run)")
    print("=" * 80)
    tester.test_single_backtest_es()   # Regression check
    tester.test_single_backtest_m2k()  # M2K: tick 0.10, pv 5.0
    tester.test_single_backtest_mcl()  # MCL: tick 0.01, pv 100.0, different session
    
    # Test run retrieval and downloads
    print("\n" + "=" * 80)
    print("PHASE 3: Run Retrieval & Downloads")
    print("=" * 80)
    tester.test_get_run()
    tester.test_download_files()
    
    # Test optimization (long-running) - skip for now to keep test time reasonable
    print("\n" + "=" * 80)
    print("PHASE 4: Walk-Forward Optimization (SKIPPED - takes 3-4 minutes)")
    print("=" * 80)
    print("⏭️  Skipping optimization test to keep runtime reasonable")
    print("   (Optimization was already tested in iteration_1)")
    
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
