#include "GdsProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

bool readCommandLineArg(int argc, char* argv[], const std::string& key, std::string& out) {
    const std::string prefix = key + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            out = arg.substr(prefix.size());
            return true;
        }
    }
    return false;
}

void parseRuntimeArgs(int argc, char* argv[], RuntimeConfig& cfg) {
    readCommandLineArg(argc, argv, "--gds", cfg.gdsPath);
    if (cfg.gdsPath.empty()) readCommandLineArg(argc, argv, "--gds_path", cfg.gdsPath);
    readCommandLineArg(argc, argv, "--camera_image", cfg.cameraImagePath);
    std::string tmp;
    if (readCommandLineArg(argc, argv, "--initial_zoom", tmp)) cfg.initialZoom = std::stod(tmp);
    // 向后兼容旧参数名。
    if (readCommandLineArg(argc, argv, "--display_scale", tmp)) cfg.initialZoom = std::stod(tmp);
    if (readCommandLineArg(argc, argv, "--scale1", tmp)) cfg.scale1 = std::stod(tmp);
    if (readCommandLineArg(argc, argv, "--scale_low", tmp)) cfg.scaleSearchLow = std::stod(tmp);
    if (readCommandLineArg(argc, argv, "--scale_high", tmp)) cfg.scaleSearchHigh = std::stod(tmp);
    if (readCommandLineArg(argc, argv, "--scale_step", tmp)) cfg.scaleSearchStep = std::stod(tmp);
    if (readCommandLineArg(argc, argv, "--render_ss", tmp)) cfg.renderSupersample = std::max(1, std::stoi(tmp));
    readCommandLineArg(argc, argv, "--result_dir", cfg.resultDir);
    if (readCommandLineArg(argc, argv, "--export_mirrored_gds", tmp)) {
        cfg.exportMirroredGds = !(tmp == "0" || tmp == "false" || tmp == "False");
    }
    if (readCommandLineArg(argc, argv, "--mirror_axis", tmp)) {
        cfg.mirrorLeftRight = !(tmp == "x" || tmp == "X");
    }
}

ZoomPanRoiSelector::ZoomPanRoiSelector(const cv::Mat& srcGray) : src(srcGray.clone()) {
    originalSize = srcGray.size();
    const int maxDim = std::max(srcGray.cols, srcGray.rows);
    if (maxDim > kMaxInteractiveDim) {
        const double s = static_cast<double>(kMaxInteractiveDim) / static_cast<double>(maxDim);
        cv::resize(srcGray, src, cv::Size(), s, s, cv::INTER_NEAREST);
        coordScaleToOriginal = 1.0 / s;
    } else {
        coordScaleToOriginal = 1.0;
    }
}

cv::Point ZoomPanRoiSelector::toImagePoint(const cv::Point& viewPt) const {
    const int dispW = std::max(1, contentRect.width);
    const int dispH = std::max(1, contentRect.height);
    const int localX = std::max(0, std::min(viewPt.x - contentRect.x, dispW - 1));
    const int localY = std::max(0, std::min(viewPt.y - contentRect.y, dispH - 1));
    const int vpW = std::min(src.cols, std::max(8, static_cast<int>(std::round(src.cols / zoom))));
    const int vpH = std::min(src.rows, std::max(8, static_cast<int>(std::round(src.rows / zoom))));
    const int maxX = std::max(0, src.cols - vpW);
    const int maxY = std::max(0, src.rows - vpH);
    const int x0 = std::max(0, std::min(static_cast<int>(std::round(pan.x)), maxX));
    const int y0 = std::max(0, std::min(static_cast<int>(std::round(pan.y)), maxY));

    int ix = x0 + static_cast<int>(std::round(static_cast<double>(localX) * vpW / dispW));
    int iy = y0 + static_cast<int>(std::round(static_cast<double>(localY) * vpH / dispH));
    if (mirrorX) ix = src.cols - 1 - ix;
    if (mirrorY) iy = src.rows - 1 - iy;
    ix = std::max(0, std::min(ix, src.cols - 1));
    iy = std::max(0, std::min(iy, src.rows - 1));
    return cv::Point(ix, iy);
}

void ZoomPanRoiSelector::render() {
    if (!cacheValid || cachedMirrorX != mirrorX || cachedMirrorY != mirrorY || cachedVis.empty()) {
        cv::cvtColor(src, cachedVis, cv::COLOR_GRAY2BGR);
        if (mirrorX || mirrorY) {
            int flipCode = 0;
            if (mirrorX && mirrorY) {
                flipCode = -1;
            } else if (mirrorX) {
                flipCode = 1;
            } else {
                flipCode = 0;
            }
            cv::flip(cachedVis, cachedVis, flipCode);
        }
        cachedMirrorX = mirrorX;
        cachedMirrorY = mirrorY;
        cacheValid = true;
    }

    cv::Rect drawRectImg;
    if (selecting) {
        const int x0 = std::min(selectStartImg.x, selectNowImg.x);
        const int y0 = std::min(selectStartImg.y, selectNowImg.y);
        const int w = std::abs(selectNowImg.x - selectStartImg.x);
        const int h = std::abs(selectNowImg.y - selectStartImg.y);
        drawRectImg = cv::Rect(x0, y0, w, h) & cv::Rect(0, 0, src.cols, src.rows);
    } else {
        drawRectImg = selectedRoi & cv::Rect(0, 0, src.cols, src.rows);
    }

    if (drawRectImg.width > 0 && drawRectImg.height > 0) {
        if (mirrorX) drawRectImg.x = src.cols - (drawRectImg.x + drawRectImg.width);
        if (mirrorY) drawRectImg.y = src.rows - (drawRectImg.y + drawRectImg.height);
    }

    const int vpW = std::min(src.cols, std::max(8, static_cast<int>(std::round(src.cols / zoom))));
    const int vpH = std::min(src.rows, std::max(8, static_cast<int>(std::round(src.rows / zoom))));
    const int maxX = std::max(0, src.cols - vpW);
    const int maxY = std::max(0, src.rows - vpH);
    const int x0 = std::max(0, std::min(static_cast<int>(std::round(pan.x)), maxX));
    const int y0 = std::max(0, std::min(static_cast<int>(std::round(pan.y)), maxY));
    const cv::Rect viewport(x0, y0, vpW, vpH);

    cv::Mat viewCropped = cachedVis(viewport);
    cv::Mat canvas(displaySize, CV_8UC3, cv::Scalar(0, 0, 0));
    const double sxFit = static_cast<double>(displaySize.width) / vpW;
    const double syFit = static_cast<double>(displaySize.height) / vpH;
    const double fitScale = std::min(sxFit, syFit);
    const int fitW = std::max(1, static_cast<int>(std::round(vpW * fitScale)));
    const int fitH = std::max(1, static_cast<int>(std::round(vpH * fitScale)));
    const int offX = (displaySize.width - fitW) / 2;
    const int offY = (displaySize.height - fitH) / 2;
    contentRect = cv::Rect(offX, offY, fitW, fitH);
    cv::Mat resized;
    cv::resize(viewCropped, resized, cv::Size(fitW, fitH), 0, 0, cv::INTER_NEAREST);
    resized.copyTo(canvas(contentRect));
    cv::Mat& view = canvas;

    if (drawRectImg.width > 0 && drawRectImg.height > 0) {
        const cv::Rect clipped = drawRectImg & viewport;
        if (clipped.width > 0 && clipped.height > 0) {
            const double sx = static_cast<double>(contentRect.width) / vpW;
            const double sy = static_cast<double>(contentRect.height) / vpH;
            cv::Rect drawOnView(contentRect.x + static_cast<int>(std::round((clipped.x - x0) * sx)),
                                contentRect.y + static_cast<int>(std::round((clipped.y - y0) * sy)),
                                std::max(1, static_cast<int>(std::round(clipped.width * sx))),
                                std::max(1, static_cast<int>(std::round(clipped.height * sy))));
            cv::rectangle(view, drawOnView, cv::Scalar(0, 255, 255), 2);
        }
    }

    cv::putText(view,
                "Left drag: ROI, Right/Middle drag: pan, Wheel/+/-: zoom, WASD: pan, M/N: mirror, Enter/Esc",
                cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::putText(view, "zoom=" + std::to_string(zoom), cv::Point(10, 48), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
    cv::imshow(winName, view);
}

void ZoomPanRoiSelector::mouseCallback(int event, int x, int y, int flags, void* userdata) {
    auto* self = static_cast<ZoomPanRoiSelector*>(userdata);
    if (self) self->onMouse(event, x, y, flags);
}

void ZoomPanRoiSelector::onMouse(int event, int x, int y, int flags) {
    if (event == cv::EVENT_MOUSEWHEEL) {
        const cv::Point anchorView(x, y);
        const cv::Point anchorImgBefore = toImagePoint(anchorView);
        const double oldZoom = zoom;
        const int delta = cv::getMouseWheelDelta(flags);
        if (delta > 0) {
            zoom = std::min(kMaxZoom, zoom * 1.1);
        } else if (delta < 0) {
            zoom = std::max(kMinZoom, zoom / 1.1);
        }
        if (std::abs(zoom - oldZoom) > 1e-12) {
            const int dispW = std::max(1, contentRect.width);
            const int dispH = std::max(1, contentRect.height);
            const int localX = std::max(0, std::min(anchorView.x - contentRect.x, dispW - 1));
            const int localY = std::max(0, std::min(anchorView.y - contentRect.y, dispH - 1));
            const int vpW = std::min(src.cols, std::max(8, static_cast<int>(std::round(src.cols / zoom))));
            const int vpH = std::min(src.rows, std::max(8, static_cast<int>(std::round(src.rows / zoom))));

            int targetX = anchorImgBefore.x;
            int targetY = anchorImgBefore.y;
            if (mirrorX) targetX = src.cols - 1 - targetX;
            if (mirrorY) targetY = src.rows - 1 - targetY;

            const double rx = static_cast<double>(localX) / dispW;
            const double ry = static_cast<double>(localY) / dispH;
            pan.x = targetX - rx * vpW;
            pan.y = targetY - ry * vpH;
        }
        dirty = true;
        return;
    }
    if (event == cv::EVENT_MBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
        panning = true;
        panStartMouse = cv::Point(x, y);
        panStartOffset = pan;
        return;
    }
    if (event == cv::EVENT_MOUSEMOVE && panning) {
        pan = panStartOffset - cv::Point2d((x - panStartMouse.x) / zoom, (y - panStartMouse.y) / zoom);
        dirty = true;
        return;
    }
    if (event == cv::EVENT_MBUTTONUP || event == cv::EVENT_RBUTTONUP) {
        panning = false;
        return;
    }
    if (event == cv::EVENT_LBUTTONDOWN) {
        selecting = true;
        selectStartImg = toImagePoint(cv::Point(x, y));
        selectNowImg = selectStartImg;
        dirty = true;
        return;
    }
    if (event == cv::EVENT_MOUSEMOVE && selecting) {
        selectNowImg = toImagePoint(cv::Point(x, y));
        dirty = true;
        return;
    }
    if (event == cv::EVENT_LBUTTONUP && selecting) {
        selecting = false;
        selectNowImg = toImagePoint(cv::Point(x, y));
        const int x0 = std::min(selectStartImg.x, selectNowImg.x);
        const int y0 = std::min(selectStartImg.y, selectNowImg.y);
        const int w = std::abs(selectNowImg.x - selectStartImg.x);
        const int h = std::abs(selectNowImg.y - selectStartImg.y);
        selectedRoi = cv::Rect(x0, y0, w, h) & cv::Rect(0, 0, src.cols, src.rows);
        dirty = true;
    }
}

cv::Rect ZoomPanRoiSelector::run(const std::string& windowName, double initialZoom) {
    winName = windowName;
    // 保持交互直观：1.0 为全图适配，>1 为放大。
    const double startZoom = (initialZoom < 1.0) ? 1.0 : initialZoom;
    zoom = std::max(kMinZoom, std::min(kMaxZoom, startZoom));
    pan = cv::Point2d(0.0, 0.0);
    // 固定交互视窗尺寸；渲染图本身分辨率不变。
    displaySize = cv::Size(1000, 700);
    contentRect = cv::Rect(0, 0, displaySize.width, displaySize.height);
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, displaySize.width, displaySize.height);
    cv::setMouseCallback(winName, &ZoomPanRoiSelector::mouseCallback, this);
    dirty = true;
    while (true) {
        if (dirty) {
            render();
            dirty = false;
        }
        const int key = cv::waitKey(20);
        if (key == 13 || key == 10) {
            confirmed = true;
            break;
        }
        if (key == 27) {
            cancel = true;
            break;
        }
        if (key == '+' || key == '=') {
            zoom = std::min(kMaxZoom, zoom * 1.1);
            dirty = true;
        } else if (key == '-' || key == '_') {
            zoom = std::max(kMinZoom, zoom / 1.1);
            dirty = true;
        } else if (key == 'w' || key == 'W') {
            pan.y += 20.0;
            dirty = true;
        } else if (key == 's' || key == 'S') {
            pan.y -= 20.0;
            dirty = true;
        } else if (key == 'a' || key == 'A') {
            pan.x += 20.0;
            dirty = true;
        } else if (key == 'd' || key == 'D') {
            pan.x -= 20.0;
            dirty = true;
        } else if (key == 'm' || key == 'M') {
            mirrorX = !mirrorX;
            dirty = true;
        } else if (key == 'n' || key == 'N') {
            mirrorY = !mirrorY;
            dirty = true;
        }
    }
    cv::destroyWindow(winName);
    if (cancel || !confirmed) return cv::Rect();
    const int ox = static_cast<int>(std::round(selectedRoi.x * coordScaleToOriginal));
    const int oy = static_cast<int>(std::round(selectedRoi.y * coordScaleToOriginal));
    const int ow = static_cast<int>(std::round(selectedRoi.width * coordScaleToOriginal));
    const int oh = static_cast<int>(std::round(selectedRoi.height * coordScaleToOriginal));
    return cv::Rect(ox, oy, ow, oh) & cv::Rect(0, 0, originalSize.width, originalSize.height);
}

GdsProcessor::GdsProcessor(const std::string& filePath) {
    if (filePath.empty()) throw std::runtime_error("GDS path is empty");
    // 当前 gdstk 头文件签名:
    // read_gds(filename, unit, tolerance, shape_tags, error_code)
    m_library = gdstk::read_gds(filePath.c_str(), 0.0, 0.0, nullptr, nullptr);
    if (m_library.cell_array.count == 0) throw std::runtime_error("No available cell in GDS");
    m_activeCell = m_library.cell_array[0];
}

GdsProcessor::~GdsProcessor() { m_library.clear(); }

GdsProcessor::GdsBBox GdsProcessor::getBoundingBox() const {
    GdsBBox box{};
    if (m_activeCell == nullptr) return box;
    gdstk::Vec2 min{}, max{};
    m_activeCell->bounding_box(min, max);
    box.min_x = min.x;
    box.min_y = min.y;
    box.max_x = max.x;
    box.max_y = max.y;
    return box;
}

void GdsProcessor::createMirroredTopCell(bool mirrorLeftRight, const std::string& mirroredCellName) {
    if (m_library.cell_array.count == 0 || m_activeCell == nullptr) {
        throw std::runtime_error("Cannot mirror empty GDS library");
    }

    gdstk::Vec2 min{}, max{};
    m_activeCell->bounding_box(min, max);
    const double cx = 0.5 * (min.x + max.x);
    const double cy = 0.5 * (min.y + max.y);

    gdstk::Cell* mirroredCell = (gdstk::Cell*)gdstk::allocate_clear(sizeof(gdstk::Cell));
    mirroredCell->name = gdstk::copy_string(mirroredCellName.c_str(), nullptr);

    gdstk::Reference* ref = (gdstk::Reference*)gdstk::allocate_clear(sizeof(gdstk::Reference));
    ref->init(m_activeCell);
    ref->magnification = 1.0;
    ref->x_reflection = true;
    constexpr double kPi = 3.14159265358979323846;

    if (mirrorLeftRight) {
        // 左右镜像（绕 Y 轴）：x' = 2*cx - x
        ref->rotation = kPi;
        ref->origin = gdstk::Vec2{2.0 * cx, 0.0};
    } else {
        // 上下镜像（绕 X 轴）：y' = 2*cy - y
        ref->rotation = 0.0;
        ref->origin = gdstk::Vec2{0.0, 2.0 * cy};
    }

    mirroredCell->reference_array.append(ref);
    m_library.cell_array.append(mirroredCell);
    m_activeCell = mirroredCell;
}

void GdsProcessor::saveLibraryAsGds(const std::string& outputPath) const {
    const gdstk::ErrorCode err = m_library.write_gds(outputPath.c_str(), 0, nullptr);
    if (err != gdstk::ErrorCode::NoError) {
        throw std::runtime_error("Failed to write mirrored GDS file");
    }
}

cv::Mat GdsProcessor::renderToImageByPixelSize(double pixelSizeUm, const std::string& outputPath,
                                               int supersample) const {
    if (m_activeCell == nullptr) throw std::runtime_error("GDS is empty, cannot render");
    if (pixelSizeUm <= 0.0) throw std::runtime_error("pixelSizeUm must be > 0");
    supersample = std::max(1, supersample);

    gdstk::Cell* cell = m_activeCell;
    gdstk::Vec2 min{}, max{};
    cell->bounding_box(min, max);
    const int imgW = std::max(1, static_cast<int>(std::ceil((max.x - min.x) / pixelSizeUm)));
    const int imgH = std::max(1, static_cast<int>(std::ceil((max.y - min.y) / pixelSizeUm)));
    const int workW = imgW * supersample;
    const int workH = imgH * supersample;
    cv::Mat work = cv::Mat::zeros(workH, workW, CV_8UC1);

    gdstk::Array<gdstk::Polygon*> polys = {};
    cell->get_polygons(true, true, -1, false, 0, polys);
    for (uint64_t i = 0; i < polys.count; ++i) {
        gdstk::Polygon* poly = polys[i];
        std::vector<cv::Point> pts;
        pts.reserve(poly->point_array.count);
        for (uint64_t j = 0; j < poly->point_array.count; ++j) {
            const int px =
                static_cast<int>(std::round((poly->point_array[j].x - min.x) / pixelSizeUm * supersample));
            const int py =
                static_cast<int>(std::round((max.y - poly->point_array[j].y) / pixelSizeUm * supersample));
            pts.emplace_back(px, py);
        }
        if (pts.size() >= 3) cv::fillPoly(work, std::vector<std::vector<cv::Point>>{pts}, 255);
    }
    for (uint64_t i = 0; i < polys.count; ++i) {
        polys[i]->clear();
        free_allocation(polys[i]);
    }
    polys.clear();
    cv::Mat img;
    if (supersample > 1) {
        cv::resize(work, img, cv::Size(imgW, imgH), 0, 0, cv::INTER_AREA);
    } else {
        img = work;
    }
    cv::imwrite(outputPath, img);
    return img;
}

cv::Rect GdsProcessor::selectRoiWithZoomPan(const cv::Mat& srcGray, const std::string& windowName,
                                            double initialZoom) {
    ZoomPanRoiSelector selector(srcGray);
    return selector.run(windowName, initialZoom);
}

cv::Mat GdsProcessor::saveRoiImage(const cv::Mat& srcGray, const cv::Rect& roi, const std::string& outputPath) {
    const cv::Rect validRoi = roi & cv::Rect(0, 0, srcGray.cols, srcGray.rows);
    if (validRoi.width <= 0 || validRoi.height <= 0) throw std::runtime_error("ROI is empty or out of bounds");
    cv::Mat roiImage = srcGray(validRoi).clone();
    cv::imwrite(outputPath, roiImage);
    return roiImage;
}

cv::Point2d GdsProcessor::roiCenterToGds(const cv::Rect& roiOnGdsImage, double pixelSizeUm) const {
    const GdsBBox box = getBoundingBox();
    const double centerPxX = roiOnGdsImage.x + roiOnGdsImage.width * 0.5;
    const double centerPxY = roiOnGdsImage.y + roiOnGdsImage.height * 0.5;
    return cv::Point2d(box.min_x + centerPxX * pixelSizeUm, box.max_y - centerPxY * pixelSizeUm);
}

GdsProcessor::RegistrationResult GdsProcessor::registerRoiToCamera(
    const cv::Mat& roiTemplateGray, const cv::Mat& cameraGray, const cv::Rect& roiOnGdsImage, double scale1,
    double scaleSearchLow, double scaleSearchHigh, double scaleSearchStep) const {
    RegistrationResult out;
    if (roiTemplateGray.empty() || cameraGray.empty() || scaleSearchStep <= 0.0) return out;

    cv::Mat cameraEq;
    cv::equalizeHist(cameraGray, cameraEq);
    cv::Mat cameraEdge;
    cv::Canny(cameraEq, cameraEdge, 60, 180);
    cv::Mat cameraEqSmall, cameraEdgeSmall;
    cv::resize(cameraEq, cameraEqSmall, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(cameraEdge, cameraEdgeSmall, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    double bestScore = -std::numeric_limits<double>::infinity();
    cv::Point bestLoc(0, 0);
    cv::Size bestSize;
    double bestScale2 = 1.0;
    std::vector<std::pair<double, double>> coarseRank;

    auto evaluateOneScale = [&](double scale2, bool coarseMode) -> bool {
        const int tplW = std::max(8, static_cast<int>(std::round(roiTemplateGray.cols * scale2)));
        const int tplH = std::max(8, static_cast<int>(std::round(roiTemplateGray.rows * scale2)));
        if (tplW >= cameraGray.cols || tplH >= cameraGray.rows) return false;

        cv::Mat tplResized;
        cv::resize(roiTemplateGray, tplResized, cv::Size(tplW, tplH), 0, 0, cv::INTER_LINEAR);
        cv::Mat tplEq;
        cv::equalizeHist(tplResized, tplEq);

        cv::Mat resultGray;
        cv::Mat resultEdge;

        if (coarseMode) {
            const int tplWSmall = std::max(8, static_cast<int>(std::round(tplW * 0.5)));
            const int tplHSmall = std::max(8, static_cast<int>(std::round(tplH * 0.5)));
            if (tplWSmall >= cameraEqSmall.cols || tplHSmall >= cameraEqSmall.rows) return false;
            cv::Mat tplEqSmall;
            cv::resize(tplEq, tplEqSmall, cv::Size(tplWSmall, tplHSmall), 0, 0, cv::INTER_AREA);
            cv::matchTemplate(cameraEqSmall, tplEqSmall, resultGray, cv::TM_CCOEFF_NORMED);
            cv::Mat tplEdgeSmall;
            cv::Canny(tplEqSmall, tplEdgeSmall, 60, 180);
            if (cv::countNonZero(tplEdgeSmall) > 16) {
                cv::matchTemplate(cameraEdgeSmall, tplEdgeSmall, resultEdge, cv::TM_CCOEFF_NORMED);
            } else {
                resultEdge = cv::Mat::zeros(resultGray.size(), CV_32F);
            }
        } else {
            cv::matchTemplate(cameraEq, tplEq, resultGray, cv::TM_CCOEFF_NORMED);
            cv::Mat tplEdge;
            cv::Canny(tplEq, tplEdge, 60, 180);
            if (cv::countNonZero(tplEdge) > 16) {
                cv::matchTemplate(cameraEdge, tplEdge, resultEdge, cv::TM_CCOEFF_NORMED);
            } else {
                resultEdge = cv::Mat::zeros(resultGray.size(), CV_32F);
            }
        }

        cv::Mat result = resultGray * 0.45f + resultEdge * 0.55f;
        double minVal = 0.0, maxVal = -std::numeric_limits<double>::infinity();
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        double diceScore = 0.0;
        if (!coarseMode) {
            const cv::Rect candRect(maxLoc.x, maxLoc.y, tplW, tplH);
            if (candRect.x >= 0 && candRect.y >= 0 && candRect.x + candRect.width <= cameraGray.cols &&
                candRect.y + candRect.height <= cameraGray.rows) {
                cv::Mat patch = cameraGray(candRect).clone();
                cv::Mat patchBin, tplBin;
                cv::threshold(patch, patchBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                cv::threshold(tplResized, tplBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                cv::Mat inter;
                cv::bitwise_and(patchBin, tplBin, inter);
                const double interCnt = static_cast<double>(cv::countNonZero(inter));
                const double patchCnt = static_cast<double>(cv::countNonZero(patchBin));
                const double tplCnt = static_cast<double>(cv::countNonZero(tplBin));
                if (patchCnt + tplCnt > 1e-9) diceScore = 2.0 * interCnt / (patchCnt + tplCnt);
            }
        }

        const double combinedScore = coarseMode ? maxVal : (0.60 * maxVal + 0.40 * diceScore);
        out.testedScales.push_back(scale2);
        out.testedScores.push_back(combinedScore);
        if (combinedScore > bestScore) {
            bestScore = combinedScore;
            bestLoc = maxLoc;
            bestSize = cv::Size(tplW, tplH);
            bestScale2 = scale2;
            out.bestTemplateGray = tplResized.clone();
            out.bestResponseMap = result.clone();
        }
        return true;
    };

    const double coarseStep = std::max(scaleSearchStep, 0.05);
    const double fineStep = std::max(scaleSearchStep, 1e-6);
    for (double s = scaleSearchLow; s <= scaleSearchHigh + 1e-9; s += coarseStep) {
        if (evaluateOneScale(s, true)) coarseRank.push_back({out.testedScores.back(), s});
    }

    std::sort(coarseRank.begin(), coarseRank.end(),
              [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                  return a.first > b.first;
              });
    const int refineTopK = std::min<int>(3, static_cast<int>(coarseRank.size()));
    for (int k = 0; k < refineTopK; ++k) {
        const double s = coarseRank[k].second;
        const double winL = std::max(scaleSearchLow, s - coarseStep);
        const double winR = std::min(scaleSearchHigh, s + coarseStep);
        for (double ss = winL; ss <= winR + 1e-9; ss += fineStep) evaluateOneScale(ss, false);
    }

    if (bestSize.width <= 0 || bestSize.height <= 0) return out;

    double secondBest = -std::numeric_limits<double>::infinity();
    const double sameScaleEps = std::max(1e-9, fineStep * 0.5);
    for (size_t i = 0; i < out.testedScores.size(); ++i) {
        if (std::abs(out.testedScales[i] - bestScale2) <= sameScaleEps) continue;
        secondBest = std::max(secondBest, out.testedScores[i]);
    }
    if (!std::isfinite(secondBest)) {
        for (size_t i = 0; i < out.testedScores.size(); ++i) {
            if (out.testedScores[i] < bestScore - 1e-12) secondBest = std::max(secondBest, out.testedScores[i]);
        }
    }

    out.ok = true;
    out.score = bestScore;
    out.peakRatio = (secondBest > 1e-12) ? (bestScore / secondBest) : 0.0;
    out.scale2 = bestScale2;
    out.refinedScale2 = bestScale2;
    out.finalScale = scale1 * bestScale2;
    out.matchedRect = cv::Rect(bestLoc.x, bestLoc.y, bestSize.width, bestSize.height);
    out.matchAreaRatio =
        static_cast<double>(out.matchedRect.area()) / static_cast<double>(cameraGray.cols * cameraGray.rows);
    out.matchedCenterPx = cv::Point2d(out.matchedRect.x + out.matchedRect.width * 0.5,
                                      out.matchedRect.y + out.matchedRect.height * 0.5);
    const cv::Point2d centerGds = roiCenterToGds(roiOnGdsImage, scale1);
    out.centerGdsX = centerGds.x;
    out.centerGdsY = centerGds.y;
    return out;
}

void GdsProcessor::saveRegistrationArtifacts(const cv::Mat& cameraGray, const RegistrationResult& reg, double scale1,
                                             const std::string& resultDir, std::ostream* log) {
    const std::filesystem::path dir(resultDir);
    const std::string explainPath = (dir / "00_matching_notes.txt").string();
    const std::string searchPath = (dir / "01_search_image_camera_gray.png").string();
    const std::string bestTplPath = (dir / "02_template_best_scale2.png").string();
    const std::string matchPath = (dir / "03_match_rect.png").string();
    const std::string overlayPath = (dir / "04_match_overlay.png").string();
    const std::string responseGrayPath = (dir / "05_response_gray.png").string();
    const std::string responseHeatPath = (dir / "06_response_heatmap.png").string();
    const std::string scoreCsvPath = (dir / "07_scale2_score_curve.csv").string();
    const std::string reportPath = (dir / "08_match_report.json").string();

    cv::imwrite(searchPath, cameraGray);

    cv::Mat vis;
    cv::cvtColor(cameraGray, vis, cv::COLOR_GRAY2BGR);
    cv::rectangle(vis, reg.matchedRect, cv::Scalar(0, 0, 255), 2);
    cv::imwrite(matchPath, vis);

    if (!reg.bestTemplateGray.empty()) cv::imwrite(bestTplPath, reg.bestTemplateGray);

    cv::Mat overlay;
    cv::cvtColor(cameraGray, overlay, cv::COLOR_GRAY2BGR);
    if (!reg.bestTemplateGray.empty() && reg.matchedRect.width > 0 && reg.matchedRect.height > 0 &&
        reg.matchedRect.x >= 0 && reg.matchedRect.y >= 0 && reg.matchedRect.x + reg.matchedRect.width <= overlay.cols &&
        reg.matchedRect.y + reg.matchedRect.height <= overlay.rows) {
        cv::Mat roi = overlay(reg.matchedRect);
        cv::Mat tplColor;
        cv::cvtColor(reg.bestTemplateGray, tplColor, cv::COLOR_GRAY2BGR);
        cv::Mat blend;
        cv::addWeighted(roi, 0.45, tplColor, 0.55, 0.0, blend);
        blend.copyTo(roi);
        cv::rectangle(overlay, reg.matchedRect, cv::Scalar(0, 255, 255), 2);
    }
    cv::imwrite(overlayPath, overlay);

    {
        std::ofstream scoreCsv(scoreCsvPath);
        scoreCsv << "scale2,score\n";
        for (size_t i = 0; i < reg.testedScales.size() && i < reg.testedScores.size(); ++i) {
            scoreCsv << std::fixed << std::setprecision(6) << reg.testedScales[i] << "," << reg.testedScores[i]
                     << "\n";
        }
    }

    if (!reg.bestResponseMap.empty()) {
        cv::Mat responseNorm;
        cv::normalize(reg.bestResponseMap, responseNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(responseGrayPath, responseNorm);
        cv::Mat responseHeat;
        cv::applyColorMap(responseNorm, responseHeat, cv::COLORMAP_JET);
        cv::imwrite(responseHeatPath, responseHeat);
    }

    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "{\n";
    report << "  \"scale1\": " << scale1 << ",\n";
    report << "  \"scale2\": " << reg.scale2 << ",\n";
    report << "  \"refined_scale2\": " << reg.refinedScale2 << ",\n";
    report << "  \"final_scale\": " << reg.finalScale << ",\n";
    report << "  \"score\": " << reg.score << ",\n";
    report << "  \"peak_ratio\": " << reg.peakRatio << ",\n";
    report << "  \"match_area_ratio\": " << reg.matchAreaRatio << ",\n";
    report << "  \"tested_scale_count\": " << reg.testedScales.size() << ",\n";
    report << "  \"matched_rect\": {\"x\": " << reg.matchedRect.x << ", \"y\": " << reg.matchedRect.y
           << ", \"w\": " << reg.matchedRect.width << ", \"h\": " << reg.matchedRect.height << "},\n";
    report << "  \"matched_center_px\": {\"x\": " << reg.matchedCenterPx.x << ", \"y\": " << reg.matchedCenterPx.y
           << "},\n";
    report << "  \"mapped_gds_center\": {\"x\": " << reg.centerGdsX << ", \"y\": " << reg.centerGdsY << "}\n";
    report << "}\n";
    std::ofstream reportFile(reportPath);
    reportFile << report.str();

    std::ofstream explain(explainPath);
    explain << "配准说明\n";
    explain << "1) 模板图: 02_template_best_scale2.png\n";
    explain << "2) 搜索图: 01_search_image_camera_gray.png\n";
    explain << "3) 搜索范围: 全图扫描\n";
    explain << "4) 匹配分数: " << reg.score << "\n";
    explain << "5) 尺度: scale2=" << reg.scale2 << "\n";

    if (log) {
        *log << "说明: " << explainPath << "\n";
        *log << "搜索图: " << searchPath << "\n";
        *log << "模板图: " << bestTplPath << "\n";
        *log << "框选图: " << matchPath << "\n";
        *log << "叠加图: " << overlayPath << "\n";
        *log << "响应灰度图: " << responseGrayPath << "\n";
        *log << "响应热力图: " << responseHeatPath << "\n";
        *log << "尺度分数CSV: " << scoreCsvPath << "\n";
        *log << "匹配报告: " << reportPath << "\n";
    }
}
