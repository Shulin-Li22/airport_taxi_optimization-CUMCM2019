import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class TaxiTripProcessor {

    public static void main(String[] args) {
        String filePath = "../data/raw/taxi_gps_data.txt";
        String outputFilePath = "../data/processed/Taxi_Trips.csv";

        // 定义机场经纬度范围
        double lonMin = 113.77003, lonMax = 113.83039;
        double latMin = 22.61773, latMax = 22.66807;

        // 定义日期格式
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        try (BufferedReader br = new BufferedReader(new FileReader(filePath));
             BufferedWriter bw = new BufferedWriter(new FileWriter(outputFilePath))) {

            String line;
            String currentLicensePlate = null;
            List<String[]> currentTripData = new ArrayList<>();

            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                String licensePlate = data[0];
                String datetime = "2013-10-22 " + data[1];
                double longitude = Double.parseDouble(data[2]);
                double latitude = Double.parseDouble(data[3]);
                int isOccupied = Integer.parseInt(data[4]);

                // 如果车牌号改变或遇到文件结束，则处理上一个车牌号的行程数据
                if (currentLicensePlate != null && !licensePlate.equals(currentLicensePlate)) {
                    processTripData(currentTripData, dateFormat, lonMin, lonMax, latMin, latMax, bw);
                    currentTripData.clear();
                }

                // 更新当前车牌号，并将数据加入当前车的行程记录
                currentLicensePlate = licensePlate;
                currentTripData.add(new String[]{licensePlate, datetime, data[2], data[3], data[4]});
            }

            // 处理最后一个车牌号的数据
            if (!currentTripData.isEmpty()) {
                processTripData(currentTripData, dateFormat, lonMin, lonMax, latMin, latMax, bw);
            }

            System.out.println("Processing completed. Results saved to " + outputFilePath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processTripData(List<String[]> records, SimpleDateFormat dateFormat,
                                        double lonMin, double lonMax, double latMin, double latMax,
                                        BufferedWriter bw) throws Exception {
        Collections.sort(records, Comparator.comparing(o -> o[1]));

        String[] tripStart = null;
        boolean isInTrip = false;

        for (int i = 0; i < records.size(); i++) {
            String[] currentRecord = records.get(i);
            Date currentTime = dateFormat.parse(currentRecord[1]);
            double currentLon = Double.parseDouble(currentRecord[2]);
            double currentLat = Double.parseDouble(currentRecord[3]);
            int isOccupied = Integer.parseInt(currentRecord[4]);

            // 判断当前记录是否在机场范围内
            boolean inAirport = (lonMin <= currentLon && currentLon <= lonMax) &&
                                (latMin <= currentLat && currentLat <= latMax);

            if (isOccupied == 1) {
                if (!isInTrip && inAirport) {
                    // 如果车辆在机场范围内并且载客，开始计时
                    tripStart = currentRecord;
                    isInTrip = true;
                } else if (isInTrip) {
                    Date startTime = dateFormat.parse(tripStart[1]);
                    long timeDiff = (currentTime.getTime() - startTime.getTime()) / 1000;

                    // 判断是否为有效的机场出发行程
                    if (timeDiff > 1200 && inAirport) {
                        tripStart = currentRecord; // 更新行程起点
                    }

                    // 判断是否到达目的地
                    if (timeDiff > 300 && !inAirport) {
                        bw.write(String.join(",", tripStart[0], tripStart[1], currentRecord[1],
                                             tripStart[2], tripStart[3], currentRecord[2], currentRecord[3]));
                        bw.newLine();
                        isInTrip = false;
                        tripStart = null;
                    }
                }
            }
        }
    }
}
