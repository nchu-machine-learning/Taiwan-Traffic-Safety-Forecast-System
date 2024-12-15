<template>
    <div class="taiwan-map" ref="map">
        <div id="map">
            <svg id="svg" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
    </div>
    <div class="shop-list">
        <h1>{{ h1 }}</h1>
        <h2>{{ h2 }}</h2>
    </div>
    <!-- <header>
        <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="125" height="125" />

        <div class="wrapper">
            <HelloWorld msg="You did it!" />
            <nav>
                <RouterLink to="/">Home</RouterLink>
                <RouterLink to="/about">About</RouterLink>
            </nav>
        </div>
    </header> -->

    <!-- <RouterView /> -->
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue';

var h1 = ref('縣市中文');
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
    const url = '/taiwan.geojson';
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