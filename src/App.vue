<template>
    <div class="taiwan-map" ref="map">
        <div id="map">
            <svg id="svg" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
    </div>
    <div class="taiwan-form flex justify-center align-middle items-center">
        <div class="p-5 px-10 w-full grid grid-cols-1 gap-4">
            <div class="relative grid">
                <div class="absolute inset-px rounded-lg bg-tw-gray" />
                <div class="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)]">
                    <div class="px-8 py-8 sm:px-10 sm:pt-10">
                        <p class="text-lg font-medium tracking-tight text-tw-dark-gray max-lg:text-center">設置參數</p>
                        <div class="mt-2 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-2">
                            <div>
                                <label for="city_name" class="block text-sm/6 font-medium text-tw-dark-gray">縣市</label>
                                <div class="mt-2">
                                    <input v-model="h1" type="text" name="city_name" id="city_name"
                                        class="block w-full h-8 rounded-md bg-white px-3 py-1.5 text-base text-gray-900 outline outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline focus:outline-2 focus:-outline-offset-2 focus:outline-tw-yellow sm:text-sm/6"
                                        disabled />
                                </div>
                            </div>

                            <div>
                                <label for="last-name" class="block text-sm/6 font-medium text-tw-dark-gray">模型</label>
                                <div class="mt-2 bg-white rounded-md px-2">
                                    <select id="model_name" name="model_name" class="block w-full h-8 rounded-md py-1.5 text-base text-gray-900 placeholder:text-gray-400 focus:outline-none sm:text-sm/6">
                                        <option value="">
                                            請選擇模型
                                        </option>
                                        <option value="GPT">
                                            GPT
                                        </option>
                                        <option value="RandomForest">
                                            RandomForest
                                        </option>
                                        <option value="AdaBoost">
                                            AdaBoost
                                        </option>
                                        <option value="GradientBoost">
                                            GradientBoost
                                        </option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="mt-2">
                            <label class="text-sm/6 font-medium text-tw-dark-gray">預測 </label>
                            <label class="text-sm/6 font-medium text-tw-dark-gray">30</label>
                            <label class="text-sm/6 font-medium text-tw-dark-gray"> 天後</label>
                            <input id="num_of_days" name="num_of_days"class="w-full accent-tw-yellow" type="range" value="30" min="30" max="365"
                                oninput="this.previousElementSibling.previousElementSibling.innerText=this.value" />
                            <div class="-mt-2 flex w-full justify-between">
                                <span class="text-sm text-gray-600">30</span>
                                <span class="text-sm text-gray-600">365</span>
                            </div>
                        </div>

                        <button type="button"
                            class="mt-2 w-full rounded-md bg-tw-yellow px-3 py-2 text-sm font-semibold text-tw-dark-gray shadow-sm hover:bg-tw-yellow focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-tw-yellow"
                            @click="updateGraph">開始預測</button>
                    </div>
                </div>
                <div class="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5" />
            </div>
            <div class="relative grid">
                <div class="absolute inset-px rounded-lg bg-tw-gray" />
                <div class="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)]">
                    <div class="px-8 py-8 sm:px-10 sm:pt-10">
                        <p class="text-lg font-medium tracking-tight text-gray-950 max-lg:text-center">
                            交通受傷人數預測結果
                        </p>
                        <LineChart class="mt-2" :data="chartData" :options="chartOptions" />
                    </div>
                </div>
                <div class="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5" />
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, nextTick, reactive } from 'vue';

var h1 = ref('請點選地圖選擇縣市');
var h2 = ref('縣市英文');
const map = ref(null);

const getTaiwanMap = async () => {
    const width = (map.value.offsetWidth).toFixed();
    const height = (map.value.offsetHeight).toFixed();

    // 判斷螢幕寬度，給不同放大值
    let mercatorScale;
    const w = window.screen.width;
    if (w > 1366) {
        mercatorScale = 11000;
    } else if (w <= 1366 && w > 480) {
        mercatorScale = 9000;
    } else {
        mercatorScale = 6000;
    }

    // d3：svg path 產生器
    var path = await d3.geo.path().projection(
        // !important 座標變換函式
        d3.geo
            .mercator()
            .center([121, 24])
            .scale(mercatorScale)
            .translate([width / 2, height / 2.5])
    );

    // 讓 d3 抓 svg，並寫入寬高
    const svg = await d3.select('#svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    // 讓 d3 抓 GeoJSON 檔，並寫入 path 的路徑
    const url = '/assets/taiwan.geojson';
    await d3.json(url, (error, geometry) => {
        if (error) throw error;
        svg
            .selectAll('path')
            .data(geometry.features)
            .enter().append('path')
            .attr('d', path)
            .attr({
                // 設定id，為了click時加class用
                id: (d) => 'city' + d.properties.COUNTYCODE
            })
            .on('click', d => {
                h1.value = d.properties.COUNTYNAME; // 換中文名
                h2.value = d.properties.COUNTYENG; // 換英文名
                // 有 .active 存在，就移除 .active
                if (document.querySelector('.active')) {
                    document.querySelector('.active').classList.remove('active');
                }
                // 被點擊的縣市加上 .active
                document.getElementById('city' + d.properties.COUNTYCODE).classList.add('active');
            });
    });
    return svg;
};

onMounted(async () => {
    await nextTick();
    getTaiwanMap();
});
</script>

<script>
import LineChart from './components/LineChart.vue';
import axios from 'axios';

export default {
    data() {
        return {
            chartData: {
                labels: [],
                datasets: [],
            },
            chartOptions: {
                responsive: true,
                pointRadius: 0,
                borderWidth: 1,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: '受傷人數'
                        },
                        ticks: {
                            precision: 0
                        }
                    }
                },
                spanGaps: true,
            },
        };
    },
    components: {
        LineChart,
    },

    methods: {
        updateGraph() {
            axios.post('http://127.0.0.1:8000/api', {
                city_name: document.getElementById('city_name').value,
                model_name: document.getElementById('model_name').value,
                num_of_days: document.getElementById('num_of_days').value
            })
            .then(response => {
                // 處理 labels
                var labels = []
                var startDate = new Date(response.data['training_starting_date']);
                var totalDays = 30 + response.data['forecast_result'].length;
                for(var i = 0; i < totalDays; i++) {
                    var currentDate = new Date(startDate);
                    currentDate.setDate(currentDate.getDate() + i);
                    labels.push(currentDate.toISOString().split('T')[0]); // 格式為 'YYYY-MM-DD'
                }

                // 處理 datasets
                var training_data = {};
                training_data['label'] = "訓練資料";
                training_data['borderColor'] = "#0284c7";
                training_data['data'] = [];
                for(var i = response.data['training_data'].length - 30; i < response.data['training_data'].length; i++) {
                    training_data['data'].push(response.data['training_data'][i])
                }
                for(var i = 0; i < 30; i++) {
                    training_data['data'].push(null)
                }
                var forecast_result = {};
                forecast_result['label'] = "預測結果";
                forecast_result['borderColor'] = "#ea580c";
                forecast_result['data'] = [];
                for(var i = 0; i < 29; i++) {
                    forecast_result['data'].push(null)
                }
                forecast_result['data'].push(response.data['training_data'][response.data['training_data'].length - 1])
                for(var i = 0; i < response.data['forecast_result'].length; i++) {
                    forecast_result['data'].push(response.data['forecast_result'][i])
                }

                this.chartData = {
                    labels: labels,
                    datasets: [
                        training_data,
                        forecast_result
                    ],
                };
            })
            .catch(error => {
                alert(error);
            });
        },
    },
};
</script>