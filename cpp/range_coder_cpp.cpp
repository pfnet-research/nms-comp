#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cstdint>
#include <fstream>
#include <string>

namespace py = pybind11;
using namespace std;
using namespace Eigen;

typedef uint64_t u8;
typedef uint32_t u4;
typedef uint16_t u2;
typedef uint8_t u1;
typedef Array<float,Dynamic,Dynamic,RowMajor> ArrXXf;
typedef Array<float,1,Dynamic,RowMajor> ArrXf;
typedef Array<u4,1,Dynamic, RowMajor> ArrXu;


class Encoder {
private:
    FILE* fp;
    u4 l;
    u4 r;
    u1 buffer;
    u8 carry;
    u8 carry0;
    bool start;

    ArrXu make_q(const ArrXf& p, const u4 _r) {
        ArrXu q;
        if (_r == 0)
            q = (0x100000000 * p + 0.5).cast<u4>();
        else
            q = (_r * p + 0.5).cast<u4>();

        q += (q == 0).cast<u4>();

        ArrXf::Index maxId;
        p.maxCoeff(&maxId);
        q[maxId] += _r - q.sum();

        return q;
    }

    void step(const ArrXf& p, const u4 t) {
        ArrXu q = make_q(p, r);
        u4 old_l = l;
        l += q.block(0,0,1,t).sum();
        if (l < old_l)
            overflow_save();
        r = q(0,t);

        while (r < 0x1000000) {
            if (l >> 24 == 0xff) {
                carry++;
            } else {
                normal_save();
            }
            l <<= 8;
            r <<= 8;
        }
    }

    void overflow_save() {
        buffer++;
        for (auto i=0; i!=carry; i++) {
            save_buffer();
            buffer = 0;
        }
        carry = 0;
    }

    void normal_save() {
        save_buffer();
        for (auto i=0; i!=carry; i++) {
            save_0xff();
        }
        buffer = u1(l >> 24);
        carry = 0;
    }

    void save_buffer() {
        if (start) {
            start = false;
        } else if (buffer == 0) {
            carry0++;
        } else {
            for (auto i=0; i!=carry0; i++) {
                fputc(0x00, fp);
            }
            carry0 = 0;
            fputc(buffer, fp);
        }
    }

    void save_0xff() {
        for (auto i=0; i!=carry0; i++) {
            fputc(0x00, fp);
        }
        carry0 = 0;
        fputc(0xff, fp);
    }

public:
    Encoder(const u2 height, const u2 width, const string& filename) {
        init(height, width, filename);
    }

    void init(const u2 height, const u2 width, const string& filename) {
        fp = fopen(filename.c_str(), "wb");
        l = 0;
        r = 0;
        buffer = 0xff;
        carry = 0;
        carry0 = 0;
        start = true;
        fputc(u1(height >> 8), fp);
        fputc(u1(height), fp);
        fputc(u1(width >> 8), fp);
        fputc(u1(width), fp);
    }

    void call(const ArrXXf& p_vec, const ArrXu& t_vec) {
        for (auto i=0; i!=t_vec.size(); i++) {
            step(p_vec.row(i), t_vec(i));
        }
    }

    void finish() {
        while (l != 0) {
            if (-l < r) {
                overflow_save();
                break;
            } else {
                normal_save();
                l <<= 8;
                r <<= 8;
            }
        }
        normal_save();
        fclose(fp);
    }
};

class Decoder {
private:
    FILE* fp;
    u4 d;
    u4 r;

    ArrXu make_q(const ArrXf& p, const u4 _r) {
        ArrXu q;
        if (_r == 0)
            q = (0x100000000 * p + 0.5).cast<u4>();
        else
            q = (_r * p + 0.5).cast<u4>();

        q += (q == 0).cast<u4>();

        ArrXf::Index maxId;
        p.maxCoeff(&maxId);
        q[maxId] += _r - q.sum();

        return q;
    }

    u4 step(const ArrXf& p) {
        while (r < 0x1000000 && r != 0) {
            d <<= 8;
            d |= _fgetc();
            r <<= 8;
        }

        ArrXu q = make_q(p, r);

        u4 idx = -1;
        u4 sum_q = 0;
        for (auto i=0; i!=q.size()-1; i++) {
            sum_q += q[i];
            if (sum_q > d) {
                idx = i;
                d -= sum_q - q[i];
                r = q[i];
                break;
            }
        }

        if (idx == -1) {
            idx = q.size()-1;
            d -= sum_q;
            r = q(idx);
        }

        return idx;
    }

    u1 _fgetc() {
        int c;
        if ((c = fgetc(fp)) == EOF)
            c = 0;
        return u1(c);
    }

public:
    u2 height;
    u2 width;

    Decoder(const string& filename) {
        init(filename);
    }

    void init(const string& filename) {
        fp = fopen(filename.c_str(), "rb");
        d = 0;
        r = 0;
        height = fgetc(fp) * 0x100 + fgetc(fp);
        width = fgetc(fp) * 0x100 + fgetc(fp);
        for (auto i=0; i!=4; i++) {
            d <<= 8;
            d |= _fgetc();
        }
    }

    ArrXu call(const ArrXXf& p_vec) {
        ArrXu reconst = ArrXu::Zero(p_vec.rows());
        for (auto i=0; i!=p_vec.rows(); i++)
            reconst(i) = step(p_vec.row(i));
        return reconst;
    }

    void finish() {
        fclose(fp);
    }

};


PYBIND11_MODULE(range_coder_cpp, m) {
	py::class_<Encoder>(m, "Encoder")
    .def(py::init<const u2, const u2, const string&>(), py::arg("height"), py::arg("width"), py::arg("filename") = "_test.bin")
    .def("init", &Encoder::init, py::arg("height"), py::arg("width"), py::arg("filename") = "_test.bin")
    .def("call", &Encoder::call)
    .def("finish", &Encoder::finish)
    ;

	py::class_<Decoder>(m, "Decoder")
    .def(py::init<const string&>(), py::arg("filename") = "_test.bin")
    .def("init", &Decoder::init, py::arg("filename") = "_test.bin")
    .def("call", &Decoder::call, py::return_value_policy::reference_internal)
    .def("finish", &Decoder::finish)
    .def_readonly("height", &Decoder::height)
    .def_readonly("width", &Decoder::width)
    ;
}